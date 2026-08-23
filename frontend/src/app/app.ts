import { Component, ElementRef, ViewChild, computed, inject, signal, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpEventType } from '@angular/common/http';

export interface ActionItem {
  task: string;
  assignee: string;
  deadline?: string;
  priority: 'High' | 'Medium' | 'Low' | string;
  completed: boolean;
}

export interface Decision {
  topic: string;
  details: string;
}

export interface TranscriptSegment {
  timestamp: string;
  start: number;
  end: number;
  speaker: string;
  text: string;
}

export interface MeetingInsights {
  duration_formatted: string;
  duration_seconds: number;
  word_count: number;
  action_items_count: number;
  decisions_count: number;
}

export interface SentimentData {
  overall: string;
  score: number;
  tone: string;
  topics: string[];
}

export interface ChatMessage {
  sender: 'user' | 'ai';
  text: string;
  timestamp: string;
}

export interface SavedMeeting {
  id: string;
  title: string;
  date: string;
  duration: string;
  summary: string;
  keyPoints: string[];
  decisions: Decision[];
  actionItems: ActionItem[];
  segments: TranscriptSegment[];
  insights: MeetingInsights;
  sentiment: SentimentData;
}

export type IngestionMode = 'upload' | 'record';
export type ProcessingStage = 'idle' | 'uploading' | 'transcribing' | 'completed' | 'error';
export type ActiveTab = 'overview' | 'tasks' | 'chat' | 'transcript' | 'history';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App implements OnDestroy {
  private http = inject(HttpClient);
  readonly backendUrl = 'http://127.0.0.1:8000/process';
  readonly chatUrl = 'http://127.0.0.1:8000/chat';

  // Navigation & UI State
  ingestionMode = signal<IngestionMode>('upload');
  activeTab = signal<ActiveTab>('overview');

  // File state
  selectedFile = signal<File | null>(null);
  fileSizeFormatted = signal<string>('');
  isDragging = signal<boolean>(false);
  audioPreviewUrl = signal<string | null>(null);

  // Live Microphone Recording State
  isRecording = signal<boolean>(false);
  recordingSeconds = signal<number>(0);
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private recordingInterval: any = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private animationFrameId: number | null = null;
  @ViewChild('visualizerCanvas') visualizerCanvas!: ElementRef<HTMLCanvasElement>;

  // Processing state
  uploadProgress = signal<number>(0);
  processingStage = signal<ProcessingStage>('idle');
  statusMessage = signal<string>('');
  errorMessage = signal<string>('');

  // Results state
  transcript = signal<string>('');
  summary = signal<string>('');
  keyPoints = signal<string[]>([]);
  decisions = signal<Decision[]>([]);
  actionItems = signal<ActionItem[]>([]);
  segments = signal<TranscriptSegment[]>([]);
  insights = signal<MeetingInsights | null>(null);
  sentiment = signal<SentimentData | null>(null);

  // Search & Filter state
  searchQuery = signal<string>('');
  priorityFilter = signal<string>('all');

  filteredSegments = computed(() => {
    const q = this.searchQuery().trim().toLowerCase();
    const segs = this.segments();
    if (!q) return segs;
    return segs.filter(s => 
      s.text.toLowerCase().includes(q) || 
      s.speaker.toLowerCase().includes(q) || 
      s.timestamp.toLowerCase().includes(q)
    );
  });

  filteredActionItems = computed(() => {
    const filter = this.priorityFilter();
    const items = this.actionItems();
    if (filter === 'all') return items;
    return items.filter(item => item.priority.toLowerCase() === filter.toLowerCase());
  });

  // Chat State
  chatMessages = signal<ChatMessage[]>([
    {
      sender: 'ai',
      text: "👋 Hi! I'm your Meeting AI Assistant. Ask me anything about this discussion, action items, deadlines, or request a drafted email.",
      timestamp: 'Just now'
    }
  ]);
  chatInput = signal<string>('');
  isChatLoading = signal<boolean>(false);
  suggestedPrompts = signal<string[]>([
    'What are the critical deadlines?',
    'Summarize each person\'s tasks',
    'What decisions were agreed upon?',
    'Draft a team follow-up email'
  ]);

  // Saved Meeting History
  savedMeetings = signal<SavedMeeting[]>([]);

  // UI feedback
  copyFeedback = signal<string | null>(null);
  private copyTimeout: any = null;

  constructor() {
    this.loadSavedMeetingsFromStorage();
  }

  ngOnDestroy(): void {
    this.stopRecordingCleanup();
  }

  // --- File Upload Handling ---

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.handleFile(input.files[0]);
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(true);
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(false);

    if (event.dataTransfer && event.dataTransfer.files.length > 0) {
      this.handleFile(event.dataTransfer.files[0]);
    }
  }

  private handleFile(file: File): void {
    if (this.audioPreviewUrl()) {
      URL.revokeObjectURL(this.audioPreviewUrl()!);
      this.audioPreviewUrl.set(null);
    }

    const validExtensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.webm', '.mp4', '.mov', '.mkv'];
    const hasValidExt = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    const isAudioOrVideo = file.type.startsWith('audio/') || file.type.startsWith('video/') || hasValidExt;

    if (!isAudioOrVideo) {
      this.errorMessage.set(`Please upload a supported audio or video file (${validExtensions.join(', ')}).`);
      this.processingStage.set('error');
      return;
    }

    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      this.errorMessage.set('File is too large! Maximum allowed size is 100MB.');
      this.processingStage.set('error');
      return;
    }

    this.selectedFile.set(file);
    this.fileSizeFormatted.set(this.formatFileSize(file.size));
    this.uploadProgress.set(0);
    this.processingStage.set('idle');
    this.statusMessage.set('');
    this.errorMessage.set('');

    if (file.type.startsWith('audio/') || file.type.startsWith('video/')) {
      this.audioPreviewUrl.set(URL.createObjectURL(file));
    }
  }

  // --- Live Microphone Recording ---

  async startRecording(): Promise<void> {
    try {
      this.errorMessage.set('');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      this.audioChunks = [];
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
      this.mediaRecorder = new MediaRecorder(stream, { mimeType });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: mimeType });
        const recordedFile = new File([audioBlob], `recorded_meeting_${Date.now()}.${mimeType.includes('webm') ? 'webm' : 'mp4'}`, { type: mimeType });
        this.handleFile(recordedFile);
        this.stopRecordingCleanup();
      };

      this.mediaRecorder.start(250);
      this.isRecording.set(true);
      this.recordingSeconds.set(0);

      // Start recording timer
      this.recordingInterval = setInterval(() => {
        this.recordingSeconds.update(s => s + 1);
      }, 1000);

      // Start visualizer
      this.setupVisualizer(stream);

    } catch (err: any) {
      console.error('Microphone access error:', err);
      this.errorMessage.set('Could not access microphone. Please allow microphone permissions in your browser.');
      this.processingStage.set('error');
    }
  }

  stopRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.isRecording.set(false);
    }
  }

  cancelRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.onstop = null;
      this.mediaRecorder.stop();
    }
    this.stopRecordingCleanup();
    this.isRecording.set(false);
    this.recordingSeconds.set(0);
  }

  private setupVisualizer(stream: MediaStream): void {
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = this.audioContext.createMediaStreamSource(stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 64;
      source.connect(this.analyser);

      const canvas = this.visualizerCanvas?.nativeElement;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const bufferLength = this.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        if (!this.isRecording()) return;
        this.animationFrameId = requestAnimationFrame(draw);

        this.analyser!.getByteFrequencyData(dataArray);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const barWidth = (canvas.width / bufferLength) * 1.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * canvas.height;
          const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
          gradient.addColorStop(0, '#3b82f6');
          gradient.addColorStop(1, '#8b5cf6');

          ctx.fillStyle = gradient;
          ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
          x += barWidth + 1;
        }
      };

      draw();
    } catch (e) {
      console.warn('Could not initialize audio visualizer:', e);
    }
  }

  private stopRecordingCleanup(): void {
    if (this.recordingInterval) {
      clearInterval(this.recordingInterval);
      this.recordingInterval = null;
    }
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  formatRecordingTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  // --- Processing & AI Pipeline ---

  uploadFile(): void {
    const file = this.selectedFile();
    if (!file) return;

    this.errorMessage.set('');
    this.transcript.set('');
    this.summary.set('');
    this.keyPoints.set([]);
    this.decisions.set([]);
    this.actionItems.set([]);
    this.segments.set([]);
    this.insights.set(null);
    this.sentiment.set(null);
    this.searchQuery.set('');

    this.uploadProgress.set(0);
    this.processingStage.set('uploading');
    this.statusMessage.set('Uploading audio recording to server...');

    const formData = new FormData();
    formData.append('file', file, file.name);

    this.http.post<any>(this.backendUrl, formData, {
      observe: 'events',
      reportProgress: true
    }).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          const percent = Math.round((100 * event.loaded) / event.total);
          this.uploadProgress.set(percent);
          if (percent === 100) {
            this.processingStage.set('transcribing');
            this.statusMessage.set('Transcribing with Whisper & extracting intelligence...');
          }
        }

        if (event.type === HttpEventType.Response) {
          const body = event.body;

          if (!body || body.success === false) {
            this.processingStage.set('error');
            this.errorMessage.set(body?.error || body?.detail || 'An unexpected error occurred during processing.');
            return;
          }

          this.processingStage.set('completed');
          this.statusMessage.set('Meeting analyzed successfully!');
          this.activeTab.set('overview');

          this.transcript.set(body.transcript || '');
          this.summary.set(body.summary || '');
          this.keyPoints.set(body.key_points || []);
          this.decisions.set(body.decisions || []);
          this.segments.set(body.segments || []);
          this.insights.set(body.insights || null);
          this.sentiment.set(body.sentiment || null);

          if (body.structured_action_items && Array.isArray(body.structured_action_items)) {
            const mapped: ActionItem[] = body.structured_action_items.map((item: any) => ({
              task: item.task,
              assignee: item.assignee || 'Team',
              deadline: item.deadline || 'Not specified',
              priority: item.priority || 'Medium',
              completed: false
            }));
            this.actionItems.set(mapped);
          }

          // Save automatically to history
          this.saveCurrentMeetingToHistory();
        }
      },
      error: (err) => {
        console.error('Processing request error:', err);
        this.processingStage.set('error');
        const detail = err.error?.detail || err.message || 'Could not connect to backend server. Make sure FastAPI is running on http://127.0.0.1:8000';
        this.errorMessage.set(detail);
      }
    });
  }

  // --- Interactive "Ask AI" Chatbot ---

  sendChatMessage(): void {
    const text = this.chatInput().trim();
    if (!text || this.isChatLoading()) return;

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    this.chatMessages.update(msgs => [...msgs, { sender: 'user', text, timestamp: time }]);
    this.chatInput.set('');
    this.isChatLoading.set(true);

    const payload = {
      question: text,
      transcript: this.transcript(),
      summary: this.summary(),
      action_items: this.actionItems(),
      decisions: this.decisions()
    };

    this.http.post<any>(this.chatUrl, payload).subscribe({
      next: (res) => {
        this.isChatLoading.set(false);
        const replyTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        this.chatMessages.update(msgs => [
          ...msgs,
          { sender: 'ai', text: res.answer, timestamp: replyTime }
        ]);

        if (res.suggested_followups && Array.isArray(res.suggested_followups)) {
          this.suggestedPrompts.set(res.suggested_followups);
        }
      },
      error: (err) => {
        this.isChatLoading.set(false);
        this.chatMessages.update(msgs => [
          ...msgs,
          { sender: 'ai', text: 'Sorry, I encountered an issue connecting to the chat service. Please check your backend connection.', timestamp: 'Now' }
        ]);
      }
    });
  }

  sendSuggestedPrompt(prompt: string): void {
    this.chatInput.set(prompt);
    this.sendChatMessage();
  }

  // --- Speaker Rename & Edit ---

  editSpeaker(oldName: string): void {
    const newName = prompt(`Enter new name for "${oldName}":`, oldName);
    if (!newName || newName.trim() === '' || newName === oldName) return;

    this.segments.update(segs => 
      segs.map(s => s.speaker === oldName ? { ...s, speaker: newName.trim() } : s)
    );
    this.triggerCopyFeedback(`Updated speaker name to "${newName.trim()}"`);
  }

  // --- Meeting History Persistence ---

  private saveCurrentMeetingToHistory(): void {
    const title = this.selectedFile()?.name.replace(/\.[^/.]+$/, '') || `Meeting ${new Date().toLocaleDateString()}`;
    const newEntry: SavedMeeting = {
      id: `meeting_${Date.now()}`,
      title,
      date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      duration: this.insights()?.duration_formatted || '0m',
      summary: this.summary(),
      keyPoints: this.keyPoints(),
      decisions: this.decisions(),
      actionItems: this.actionItems(),
      segments: this.segments(),
      insights: this.insights()!,
      sentiment: this.sentiment()!
    };

    this.savedMeetings.update(list => [newEntry, ...list.slice(0, 9)]);
    this.syncHistoryStorage();
  }

  loadSavedMeeting(meeting: SavedMeeting): void {
    this.summary.set(meeting.summary);
    this.keyPoints.set(meeting.keyPoints);
    this.decisions.set(meeting.decisions);
    this.actionItems.set(meeting.actionItems);
    this.segments.set(meeting.segments);
    this.insights.set(meeting.insights);
    this.sentiment.set(meeting.sentiment);
    this.transcript.set(meeting.segments.map(s => `[${s.timestamp}] ${s.speaker}: ${s.text}`).join('\n') || meeting.summary);
    
    this.processingStage.set('completed');
    this.activeTab.set('overview');
    this.triggerCopyFeedback(`Loaded "${meeting.title}" from history`);
  }

  deleteSavedMeeting(id: string, event: Event): void {
    event.stopPropagation();
    this.savedMeetings.update(list => list.filter(m => m.id !== id));
    this.syncHistoryStorage();
    this.triggerCopyFeedback('Deleted meeting from history');
  }

  private syncHistoryStorage(): void {
    try {
      if (typeof window !== 'undefined' && 'localStorage' in window && window.localStorage) {
        window.localStorage.setItem('meeting_ai_history', JSON.stringify(this.savedMeetings()));
      }
    } catch {
      // Storage unavailable in SSR/test environment
    }
  }

  private loadSavedMeetingsFromStorage(): void {
    try {
      if (typeof window !== 'undefined' && 'localStorage' in window && window.localStorage) {
        const raw = window.localStorage.getItem('meeting_ai_history');
        if (raw) {
          this.savedMeetings.set(JSON.parse(raw));
        }
      }
    } catch {
      // Storage unavailable in SSR/test environment
    }
  }

  // --- Actions & Helpers ---

  toggleActionItem(index: number): void {
    this.actionItems.update(items => {
      const copy = [...items];
      if (copy[index]) {
        copy[index] = { ...copy[index], completed: !copy[index].completed };
      }
      return copy;
    });
  }

  copyToClipboard(text: string, label: string): void {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      this.triggerCopyFeedback(`Copied ${label} to clipboard!`);
    }).catch(() => {
      this.triggerCopyFeedback(`Failed to copy to clipboard`);
    });
  }

  copyAllActionItems(): void {
    const items = this.actionItems();
    if (items.length === 0) return;
    const text = items
      .map((item, i) => `${i + 1}. [${item.completed ? 'x' : ' '}] ${item.task} | Assignee: ${item.assignee} | Deadline: ${item.deadline} | Priority: ${item.priority}`)
      .join('\n');
    this.copyToClipboard(text, 'action items checklist');
  }

  downloadMeetingReport(format: 'txt' | 'md' | 'pdf'): void {
    if (format === 'pdf') {
      window.print();
      return;
    }

    const title = this.selectedFile()?.name.replace(/\.[^/.]+$/, '') || 'meeting';
    let content = `# 🎙️ MEETING-TO-ACTION AI REPORT\n`;
    content += `File: ${this.selectedFile()?.name || 'Meeting Recording'}\n`;
    if (this.insights()) {
      content += `Duration: ${this.insights()!.duration_formatted} | Words: ${this.insights()!.word_count} | Decisions: ${this.insights()!.decisions_count} | Actions: ${this.insights()!.action_items_count}\n`;
    }
    if (this.sentiment()) {
      content += `Sentiment: ${this.sentiment()!.overall} (Score: ${this.sentiment()!.score}/100) | Tone: ${this.sentiment()!.tone}\n`;
    }
    content += `\n=========================================\n`;
    content += `📝 EXECUTIVE SUMMARY\n`;
    content += `=========================================\n${this.summary()}\n\n`;

    if (this.keyPoints().length > 0) {
      content += `=========================================\n📌 KEY POINTS\n=========================================\n`;
      this.keyPoints().forEach(kp => content += `• ${kp}\n`);
      content += `\n`;
    }

    if (this.decisions().length > 0) {
      content += `=========================================\n🤝 DECISIONS\n=========================================\n`;
      this.decisions().forEach(d => content += `[${d.topic}] ${d.details}\n`);
      content += `\n`;
    }

    if (this.actionItems().length > 0) {
      content += `=========================================\n✅ ACTION ITEMS & TASKS\n=========================================\n`;
      this.actionItems().forEach((item, i) => {
        content += `${i + 1}. [${item.completed ? 'COMPLETED' : 'PENDING'}] ${item.task}\n   - Assignee: ${item.assignee}\n   - Deadline: ${item.deadline}\n   - Priority: ${item.priority}\n\n`;
      });
    }

    content += `=========================================\n📄 FULL TRANSCRIPT\n=========================================\n`;
    if (this.segments().length > 0) {
      this.segments().forEach(s => {
        content += `[${s.timestamp}] ${s.speaker}: ${s.text}\n`;
      });
    } else {
      content += `${this.transcript()}\n`;
    }

    const mime = format === 'md' ? 'text/markdown' : 'text/plain';
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title}_report.${format}`;
    a.click();
    URL.revokeObjectURL(url);
    this.triggerCopyFeedback(`Downloaded meeting report (.${format})`);
  }

  private triggerCopyFeedback(msg: string): void {
    if (this.copyTimeout) {
      clearTimeout(this.copyTimeout);
    }
    this.copyFeedback.set(msg);
    this.copyTimeout = setTimeout(() => {
      this.copyFeedback.set(null);
    }, 2500);
  }

  resetAll(): void {
    if (this.audioPreviewUrl()) {
      URL.revokeObjectURL(this.audioPreviewUrl()!);
      this.audioPreviewUrl.set(null);
    }
    this.selectedFile.set(null);
    this.fileSizeFormatted.set('');
    this.uploadProgress.set(0);
    this.processingStage.set('idle');
    this.statusMessage.set('');
    this.errorMessage.set('');
    this.transcript.set('');
    this.summary.set('');
    this.keyPoints.set([]);
    this.decisions.set([]);
    this.actionItems.set([]);
    this.segments.set([]);
    this.insights.set(null);
    this.sentiment.set(null);
    this.searchQuery.set('');
    this.activeTab.set('overview');
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}