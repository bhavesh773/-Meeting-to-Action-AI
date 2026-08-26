import numpy as np
from diarization import (
    extract_acoustic_features,
    determine_optimal_speakers,
    smooth_speaker_labels,
    detect_names_in_transcript,
    diarize_and_align
)

def test_feature_extraction():
    sr = 16000
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    
    feats = extract_acoustic_features(audio, sr)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (32,)
    assert not np.isnan(feats).any()
    print("Feature extraction test passed: 32-dim acoustic vector generated successfully.")

def test_single_speaker_clustering():
    sr = 16000
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    # Similar voice characteristics
    audio1 = 0.5 * np.sin(2 * np.pi * 150 * t)
    audio2 = 0.5 * np.sin(2 * np.pi * 152 * t)
    audio3 = 0.5 * np.sin(2 * np.pi * 149 * t)
    
    f1 = extract_acoustic_features(audio1, sr)
    f2 = extract_acoustic_features(audio2, sr)
    f3 = extract_acoustic_features(audio3, sr)
    
    embeddings = np.array([f1, f2, f3])
    k, labels = determine_optimal_speakers(embeddings, max_speakers=4)
    print(f"Single speaker test result: k={k}, labels={labels}")
    assert k == 1
    assert len(np.unique(labels)) == 1

def test_multi_speaker_clustering():
    sr = 16000
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    # Speaker 1: low pitch 110Hz male voice
    s1_a = 0.6 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)
    s1_b = 0.6 * np.sin(2 * np.pi * 115 * t) + 0.3 * np.sin(2 * np.pi * 230 * t)
    
    # Speaker 2: high pitch 340Hz female/child voice
    s2_a = 0.6 * np.sin(2 * np.pi * 340 * t) + 0.3 * np.sin(2 * np.pi * 680 * t)
    s2_b = 0.6 * np.sin(2 * np.pi * 345 * t) + 0.3 * np.sin(2 * np.pi * 690 * t)
    
    f1 = extract_acoustic_features(s1_a, sr)
    f2 = extract_acoustic_features(s1_b, sr)
    f3 = extract_acoustic_features(s2_a, sr)
    f4 = extract_acoustic_features(s2_b, sr)
    
    embeddings = np.array([f1, f2, f3, f4])
    k, labels = determine_optimal_speakers(embeddings, max_speakers=4)
    print(f"Multi-speaker test result: k={k}, labels={labels}")
    assert k == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]

def test_full_diarize_and_align():
    sr = 16000
    t1 = np.linspace(0, 3.0, int(3.0 * sr), endpoint=False)
    t2 = np.linspace(0, 3.0, int(3.0 * sr), endpoint=False)
    spk1_audio = 0.5 * np.sin(2 * np.pi * 120 * t1)
    spk2_audio = 0.5 * np.sin(2 * np.pi * 350 * t2)
    full_audio = np.concatenate([spk1_audio, spk2_audio])
    
    raw_segments = [
        {"start": 0.0, "end": 1.5, "text": "Good morning team, let's start our meeting."},
        {"start": 1.5, "end": 3.0, "text": "I will review the backend architecture."},
        {"start": 3.0, "end": 4.5, "text": "Thanks Rahul, I will handle the Angular frontend."},
        {"start": 4.5, "end": 6.0, "text": "We will finalize the deployment by Friday."}
    ]
    
    result = diarize_and_align(full_audio, sr, raw_segments)
    print("Diarize & Align Output:")
    print("Num speakers:", result["num_speakers"])
    print("Speaker stats:", result["speaker_stats"])
    for seg in result["segments"]:
        print(f"[{seg['timestamp']}] {seg['speaker']}: {seg['text']}")
        
    assert result["num_speakers"] >= 1
    assert len(result["segments"]) == 4
    assert len(result["speaker_stats"]) == result["num_speakers"]
    total_pct = sum(s["talk_time_percentage"] for s in result["speaker_stats"])
    assert 99.0 <= total_pct <= 101.0
    print("All diarization tests passed successfully!")

if __name__ == "__main__":
    test_feature_extraction()
    test_single_speaker_clustering()
    test_multi_speaker_clustering()
    test_full_diarize_and_align()
