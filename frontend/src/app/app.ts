import { Component, computed, inject, signal } from '@angular/core';
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
  text: string;
}

export interface MeetingInsights {
  duration_formatted: string;
  duration_seconds: number;
  word_count: number;
  action_items_count: number;
  decisions_count: number;
}

export type ProcessingStage = 'idle' | 'uploading' | 'transcribing' | 'completed' | 'error';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  private http = inject(HttpClient);
  readonly backendUrl = 'http://127.0.0.1:8000/process';

  // File state
  selectedFile = signal<File | null>(null);
  fileSizeFormatted = signal<string>('');
  isDragging = signal<boolean>(false);
  audioPreviewUrl = signal<string | null>(null);

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

  // Search & Filter state
  searchQuery = signal<string>('');

  filteredSegments = computed(() => {
    const q = this.searchQuery().trim().toLowerCase();
    const segs = this.segments();
    if (!q) return segs;
    return segs.filter(s => s.text.toLowerCase().includes(q) || s.timestamp.toLowerCase().includes(q));
  });

  // UI feedback
  copyFeedback = signal<string | null>(null);
  private copyTimeout: any = null;

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

    const validExtensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.webm', '.mp4', '.mov'];
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

    if (file.type.startsWith('audio/')) {
      this.audioPreviewUrl.set(URL.createObjectURL(file));
    }
  }

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
            this.statusMessage.set('Transcribing with Whisper & extracting decisions and action items...');
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

          this.transcript.set(body.transcript || '');
          this.summary.set(body.summary || '');
          this.keyPoints.set(body.key_points || []);
          this.decisions.set(body.decisions || []);
          this.segments.set(body.segments || []);
          this.insights.set(body.insights || null);

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
        content += `[${s.timestamp}] ${s.text}\n`;
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
    this.searchQuery.set('');
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}