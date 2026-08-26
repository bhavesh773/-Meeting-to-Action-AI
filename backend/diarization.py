import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import scipy.signal
import scipy.fftpack
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def format_timestamp(seconds: float) -> str:
    """Formats float seconds into mm:ss string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def extract_acoustic_features(audio_segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extracts a robust, normalized acoustic signature capturing vocal tract geometry (MFCCs 1-13),
    pitch/F0 fundamental frequency, spectral centroid, and voice dynamics.
    """
    if len(audio_segment) < 256:
        return np.zeros(32, dtype=np.float32)

    # Pre-emphasis filter
    emphasized = np.append(audio_segment[0], audio_segment[1:] - 0.97 * audio_segment[:-1])
    
    frame_length = int(0.025 * sample_rate)
    frame_step = int(0.010 * sample_rate)
    
    if len(emphasized) < frame_length:
        emphasized = np.pad(emphasized, (0, frame_length - len(emphasized)))
        
    num_frames = 1 + int(np.floor((len(emphasized) - frame_length) / frame_step))
    if num_frames <= 0:
        num_frames = 1
        
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    indices = np.clip(indices, 0, len(emphasized) - 1)
    
    frames = emphasized[indices] * np.hamming(frame_length)
    
    # FFT and Power Spectrum
    nfft = 512
    mag_frames = np.abs(np.fft.rfft(frames, nfft))
    pow_frames = (1.0 / nfft) * (mag_frames ** 2)
    pow_frames = np.maximum(pow_frames, 1e-12)
    
    # Mel Filterbank (26 filters)
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, 28)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)
    
    fbank = np.zeros((26, int(np.floor(nfft / 2 + 1))))
    for m in range(1, 27):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)
                
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.maximum(filter_banks, 1e-12)
    filter_banks = 20 * np.log10(filter_banks)
    
    # DCT to obtain MFCCs (take coefficients 1 to 13, discarding C0 energy bias)
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:14]
    
    # Delta MFCCs
    if num_frames > 1:
        delta_mfcc = np.diff(mfcc, axis=0, prepend=mfcc[0:1, :])
    else:
        delta_mfcc = np.zeros_like(mfcc)
        
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)
    delta_mean = np.mean(delta_mfcc, axis=0)[:6]
    
    # Spectral Centroid
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    centroids = np.sum(mag_frames * freqs, axis=1) / (np.sum(mag_frames, axis=1) + 1e-12)
    centroid_mean = np.mean(centroids)
    
    # Pitch / F0 estimation via Autocorrelation
    pitch_estimates = []
    min_lag = int(sample_rate / 400)  # max 400Hz
    max_lag = int(sample_rate / 65)   # min 65Hz
    for f in frames[::max(1, num_frames // 8)]:
        corr = np.correlate(f, f, mode='full')
        corr = corr[len(f)-1:]
        if len(corr) > max_lag:
            peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
            if corr[peak_lag] > 0.25 * corr[0]:
                pitch_estimates.append(sample_rate / peak_lag)
    
    pitch_mean = np.mean(pitch_estimates) if pitch_estimates else 160.0
    
    # Assemble feature vector: 13 (mfcc mean) + 13 (mfcc std) + 4 (delta) + 2 (pitch, centroid) = 32
    raw_vec = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean[:4],
        np.array([
            (pitch_mean - 150.0) / 100.0 * 2.5,
            (centroid_mean - 1500.0) / 1000.0 * 1.5
        ], dtype=np.float32)
    ])
    
    norm = np.linalg.norm(raw_vec)
    if norm > 0:
        return raw_vec / norm
    return raw_vec

def determine_optimal_speakers(embeddings: np.ndarray, max_speakers: int = 6) -> Tuple[int, np.ndarray]:
    """
    Evaluates clustering quality across speaker count candidates (1 to max_speakers)
    using Agglomerative Cosine Clustering and Silhouette analysis.
    Returns optimal speaker count and cluster labels.
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return 1, np.array([], dtype=int)
    if n_samples == 1:
        return 1, np.array([0], dtype=int)
        
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    norm_embeddings = embeddings / norms
    
    # Calculate pairwise cosine distances
    dot_products = np.dot(norm_embeddings, norm_embeddings.T)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    pairwise_distances = 1.0 - dot_products
    
    triu_idx = np.triu_indices(n_samples, k=1)
    avg_pairwise_dist = float(np.mean(pairwise_distances[triu_idx]))
    max_pairwise_dist = float(np.max(pairwise_distances[triu_idx]))
    
    if n_samples == 2:
        if avg_pairwise_dist < 0.10:
            return 1, np.array([0, 0], dtype=int)
        else:
            return 2, np.array([0, 1], dtype=int)
            
    # Single speaker check
    if avg_pairwise_dist < 0.08 and max_pairwise_dist < 0.15:
        return 1, np.zeros(n_samples, dtype=int)
    
    max_k = min(max_speakers, n_samples - 1)
    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(n_samples, dtype=int)
    
    for k in range(2, max_k + 1):
        try:
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(norm_embeddings)
            
            counts = np.bincount(labels)
            if np.min(counts) < 1:
                continue
                
            score = float(silhouette_score(norm_embeddings, labels, metric='cosine'))
            penalty = 0.03 * (k - 2)
            adjusted_score = score - penalty
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_k = k
                best_labels = labels
        except Exception:
            continue
            
    if best_score < 0.18 and avg_pairwise_dist < 0.15:
        return 1, np.zeros(n_samples, dtype=int)
        
    return best_k, best_labels

def smooth_speaker_labels(labels: np.ndarray, durations: List[float], min_turn_duration: float = 1.5) -> np.ndarray:
    """
    Applies temporal smoothing to remove rapid spurious speaker flickering
    for very short segments surrounded by the same speaker.
    """
    smoothed = np.copy(labels)
    n = len(smoothed)
    if n < 3:
        return smoothed
        
    for i in range(1, n - 1):
        if durations[i] < min_turn_duration and smoothed[i - 1] == smoothed[i + 1]:
            smoothed[i] = smoothed[i - 1]
            
    return smoothed

def detect_names_in_transcript(segments: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Scans conversational text to identify explicit self-introductions or speaker vocatives
    (e.g., 'I am Rahul', 'My name is Sarah', 'This is Priya speaking').
    """
    speaker_names: Dict[int, str] = {}
    
    intro_patterns = [
        r'\b(?:i am|i\'m|my name is|this is)\s+([A-Z][a-z]+)\b',
        r'\b([A-Z][a-z]+)\s+speaking\b',
        r'^([A-Z][a-z]+):\s*'
    ]
    
    for seg in segments:
        spk_id = seg.get("speaker_id", 0)
        text = seg.get("text", "").strip()
        
        for pattern in intro_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                cand = m.group(1).capitalize()
                if cand.lower() not in ["here", "there", "today", "now", "just", "sure", "okay", "yes", "no", "well", "team"]:
                    if spk_id not in speaker_names:
                        speaker_names[spk_id] = cand
                        break
                        
    return speaker_names

def diarize_and_align(
    audio: np.ndarray,
    sample_rate: int,
    raw_segments: List[Dict[str, Any]],
    expected_speakers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Performs speaker diarization by extracting acoustic voice embeddings for each Whisper segment,
    clustering them to identify distinct speakers, smoothing speaker transitions,
    and computing speaker talk-time distribution and metrics.
    """
    if not raw_segments or len(audio) == 0:
        return {
            "num_speakers": 0,
            "speaker_stats": [],
            "segments": []
        }
        
    embeddings = []
    durations = []
    
    for idx, seg in enumerate(raw_segments):
        start_sec = seg.get("start", 0.0)
        end_sec = seg.get("end", 0.0)
        dur = max(0.1, end_sec - start_sec)
        durations.append(dur)
        
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        
        start_sample = max(0, min(start_sample, len(audio) - 1))
        end_sample = max(start_sample + 256, min(end_sample, len(audio)))
        
        seg_audio = audio[start_sample:end_sample]
        
        if len(seg_audio) >= 256:
            feat = extract_acoustic_features(seg_audio, sample_rate)
            embeddings.append(feat)
        else:
            embeddings.append(np.zeros(32, dtype=np.float32))
            
    embeddings_matrix = np.array(embeddings, dtype=np.float32)
    
    # Determine speaker clustering
    if expected_speakers and expected_speakers > 0:
        num_spk = min(expected_speakers, len(raw_segments))
        if num_spk > 1 and len(embeddings_matrix) >= num_spk:
            clustering = AgglomerativeClustering(n_clusters=num_spk, metric='cosine', linkage='average')
            labels = clustering.fit_predict(embeddings_matrix)
        else:
            labels = np.zeros(len(raw_segments), dtype=int)
    else:
        num_spk, labels = determine_optimal_speakers(embeddings_matrix, max_speakers=6)
        
    # Smooth labels
    smoothed_labels = smooth_speaker_labels(labels, durations)
    
    # Map raw cluster IDs to sequential 1-based speaker IDs in order of speech
    speaker_id_map: Dict[int, int] = {}
    next_id = 1
    for lbl in smoothed_labels:
        if lbl not in speaker_id_map:
            speaker_id_map[lbl] = next_id
            next_id += 1
            
    formatted_segments = []
    for idx, seg in enumerate(raw_segments):
        raw_lbl = smoothed_labels[idx]
        spk_num = speaker_id_map.get(raw_lbl, 1)
        start_sec = seg.get("start", 0.0)
        end_sec = seg.get("end", 0.0)
        text_clean = seg.get("text", "").strip()
        
        # Check if text already has speaker prefix like "Alice: Hello"
        explicit_match = re.match(r'^([A-Z][a-z]+):\s*(.+)', text_clean)
        if explicit_match:
            speaker_name = explicit_match.group(1)
            text_clean = explicit_match.group(2)
        else:
            speaker_name = f"Speaker {spk_num}"
            
        formatted_segments.append({
            "timestamp": format_timestamp(start_sec),
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "speaker": speaker_name,
            "speaker_id": spk_num,
            "text": text_clean
        })
        
    # Natural conversation speaker name discovery
    discovered_names = detect_names_in_transcript(formatted_segments)
    for seg in formatted_segments:
        spk_id = seg["speaker_id"]
        if spk_id in discovered_names and seg["speaker"].startswith("Speaker "):
            seg["speaker"] = discovered_names[spk_id]
            
    # Calculate per-speaker analytics & talk-time distribution
    total_audio_duration = max(0.1, sum(durations))
    speaker_talk_time: Dict[str, float] = {}
    speaker_words: Dict[str, int] = {}
    speaker_segments_count: Dict[str, int] = {}
    
    for seg in formatted_segments:
        spk = seg["speaker"]
        dur = max(0.0, seg["end"] - seg["start"])
        words = len(seg["text"].split())
        
        speaker_talk_time[spk] = speaker_talk_time.get(spk, 0.0) + dur
        speaker_words[spk] = speaker_words.get(spk, 0) + words
        speaker_segments_count[spk] = speaker_segments_count.get(spk, 0) + 1
        
    speaker_stats = []
    # Distinct color palette tokens for frontend rendering
    color_palette = [
        {"bg": "rgba(59, 130, 246, 0.15)", "border": "#3b82f6", "text": "#60a5fa", "name": "Blue"},
        {"bg": "rgba(139, 92, 246, 0.15)", "border": "#8b5cf6", "text": "#a78bfa", "name": "Purple"},
        {"bg": "rgba(16, 185, 129, 0.15)", "border": "#10b981", "text": "#34d399", "name": "Emerald"},
        {"bg": "rgba(245, 158, 11, 0.15)", "border": "#f59e0b", "text": "#fbbf24", "name": "Amber"},
        {"bg": "rgba(236, 72, 153, 0.15)", "border": "#ec4899", "text": "#f472b6", "name": "Rose"},
        {"bg": "rgba(6, 182, 212, 0.15)", "border": "#06b6d4", "text": "#22d3ee", "name": "Cyan"}
    ]
    
    for idx, (spk, talk_sec) in enumerate(speaker_talk_time.items()):
        pct = round((talk_sec / total_audio_duration) * 100, 1)
        wcount = speaker_words.get(spk, 0)
        mins = int(talk_sec // 60)
        secs = int(talk_sec % 60)
        color = color_palette[idx % len(color_palette)]
        
        speaker_stats.append({
            "speaker": spk,
            "speaker_index": idx + 1,
            "talk_time_seconds": round(talk_sec, 1),
            "talk_time_formatted": f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s",
            "talk_time_percentage": min(100.0, pct),
            "word_count": wcount,
            "segments_count": speaker_segments_count.get(spk, 0),
            "color_theme": color
        })
        
    speaker_stats.sort(key=lambda x: x["talk_time_seconds"], reverse=True)
    
    return {
        "num_speakers": len(speaker_stats),
        "speaker_stats": speaker_stats,
        "segments": formatted_segments
    }
