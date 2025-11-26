#!/usr/bin/env python3
"""
video_to_short_clips.py

Requirements:
  - Python 3.8+
  - ffmpeg installed and in PATH
  - pip install openai ffmpeg-python python-dotenv tqdm pyscenedetect

Usage:
  python video_to_short_clips.py input.mp4 --out-dir clips --max-clips 8 --max-duration 45
"""

import os
import sys
import json
import subprocess
import argparse
from math import ceil
from tqdm import tqdm
from typing import List, Dict, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional imports if you want scene-detect fallback
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except Exception:
    SCENEDETECT_AVAILABLE = False

# OpenAI imports (if using OpenAI)
try:
    import openai
except Exception:
    openai = None

# ---------- Helpers ----------
def run_ffmpeg_trim(input_path: str, out_path: str, start: float, end: float, reencode: bool=False):
    """
    Trim a segment from input_path [start, end) into out_path.
    If reencode=False we try stream copy (fast but less precise for some codecs).
    """
    duration = end - start
    if duration <= 0:
        raise ValueError("end must be > start")

    if not reencode:
        # using -ss before -i with copy; added -c:a aac to ensure audio is preserved
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", input_path,
            "-t", f"{duration:.3f}",
            "-c:v", "copy",
            "-c:a", "aac",
            "-avoid_negative_ts", "1",
            out_path
        ]
    else:
        # re-encode for precise frame-accurate trimming
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            out_path
        ]
    subprocess.run(cmd, check=True, capture_output=True)

def load_transcript_via_openai_whisper(audio_path: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Call OpenAI whisper API to get transcript segments.
    Returns list of {start,end,text}.
    """
    if openai is None:
        raise RuntimeError("openai package not installed or not available")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    # Check if input is a video file, extract audio if needed
    if audio_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv')):
        print("Extracting audio from video...")
        audio_temp = "temp_audio.mp3"
        
        # Remove temp file if it exists
        if os.path.exists(audio_temp):
            os.remove(audio_temp)
        
        # Extract audio using ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-q:a", "9",
            audio_temp
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Audio extraction failed: {result.stderr}")
            raise RuntimeError(f"ffmpeg failed with code {result.returncode}")
        
        audio_path = audio_temp
    
    try:
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                language="en"
            )
        
        segments = []
        for seg in resp.get("segments", []):
            segments.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()})
        return segments
    finally:
        # Clean up temp audio file
        if audio_path.startswith("temp_audio"):
            try:
                os.remove(audio_path)
            except:
                pass

def call_llm_for_highlights(transcript_segments: List[Dict[str,Any]], api_key: str,
                            max_clips:int=6, max_clip_duration:int=60, min_clip_duration:int=6):
    """
    Sends a prompt to the LLM to produce JSON highlights.
    """
    if openai is None:
        raise RuntimeError("openai package not installed or not available")
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    transcript_json = json.dumps(transcript_segments, ensure_ascii=False)
    prompt = f"""
You are given a transcript of a long video as a JSON array of segments (each with start, end, text).
Pick up to {max_clips} short social-media clips (each <= {max_clip_duration}s, >= {min_clip_duration}s).
Output ONLY JSON array of objects: {{start, end, title, reason, confidence}}.
Prefer self-contained moments, avoid cutting mid-sentence. Transcript: {transcript_json}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.0,
        max_tokens=800
    )
    
    content = response.choices[0].message.content
    try:
        import re
        m = re.search(r"```(?:json)?\s*(\[.*?\])", content, re.S)
        if m:
            json_text = m.group(1)
        else:
            json_text = content
        data = json.loads(json_text)
        return data
    except Exception as e:
        print("Failed to parse LLM output as JSON:", e)
        print("Raw output:")
        print(content)
        raise

def merge_and_normalize_segments(raw_segments: List[Dict[str,Any]], max_clip_duration:int, min_clip_duration:int):
    """
    Merge overlapping/adjacent segments and adjust durations to be within min/max by expanding a bit if needed.
    Input segments must have start,end.
    """
    if not raw_segments:
        return []
    # sort
    segs = sorted(raw_segments, key=lambda s: s["start"])
    merged = []
    cur = segs[0].copy()
    for s in segs[1:]:
        if s["start"] <= cur["end"] + 1.0:  # small gap -> merge
            cur["end"] = max(cur["end"], s["end"])
            # optionally merge titles/reasons
            cur["title"] = cur.get("title","") + " | " + s.get("title","")
        else:
            merged.append(cur)
            cur = s.copy()
    merged.append(cur)

    # enforce durations and expand (center expansion)
    out = []
    for s in merged:
        dur = s["end"] - s["start"]
        if dur > max_clip_duration:
            # split into multiple parts
            n_parts = ceil(dur / max_clip_duration)
            part_len = dur / n_parts
            for i in range(n_parts):
                start = s["start"] + i*part_len
                end = min(s["start"] + (i+1)*part_len, s["end"])
                out.append({"start": start, "end": end, "title": s.get("title",""), "reason": s.get("reason",""), "confidence": s.get("confidence",1.0)})
        else:
            if dur < min_clip_duration:
                # expand by half of (min-dur) on both sides where possible
                need = min_clip_duration - dur
                half = need/2
                s_start = max(0.0, s["start"] - half)
                s_end = s["end"] + half
                s["start"], s["end"] = s_start, s_end
            out.append({"start": s["start"], "end": s["end"], "title": s.get("title",""), "reason": s.get("reason",""), "confidence": s.get("confidence",1.0)})
    return out

def scenedetect_fallback(input_path:str, max_clips:int=6):
    """Return list of scene boundaries using PySceneDetect if available."""
    if not SCENEDETECT_AVAILABLE:
        return []
    video_manager = VideoManager([input_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()
    try:
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)
        results = []
        for i,(s,e) in enumerate(scene_list):
            results.append({"start": s.get_seconds(), "end": e.get_seconds(), "title": f"Scene {i+1}", "reason":"scene change", "confidence":0.5})
        return results[:max_clips]
    finally:
        video_manager.release()

# --------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input video file")
    parser.add_argument("--out-dir", default="clips", help="output directory")
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--max-duration", type=int, default=60)
    parser.add_argument("--min-duration", type=int, default=6)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--reencode", action="store_true", help="re-encode trims for frame accuracy")
    parser.add_argument("--use-scene-fallback", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Transcribe using Whisper (OpenAI) — adjust if offline whisper used
    transcript_segments = []
    if args.api_key and openai is not None:
        try:
            print("Transcribing with Whisper...")
            transcript_segments = load_transcript_via_openai_whisper(args.input, args.api_key)
            print(f"Got {len(transcript_segments)} transcript segments.")
        except Exception as e:
            print("Transcription failed:", e)

    # 2) If no transcript, try scene detection fallback
    if (not transcript_segments) and args.use_scene_fallback:
        print("Trying scene-detection fallback...")
        transcript_segments = scenedetect_fallback(args.input, max_clips=args.max_clips)

    # 3) If we have transcript, ask LLM for highlight segments
    highlights = []
    if transcript_segments and args.api_key and openai is not None:
        try:
            print("Asking LLM for highlights...")
            raw = call_llm_for_highlights(transcript_segments, api_key=args.api_key,
                                          max_clips=args.max_clips,
                                          max_clip_duration=args.max_duration,
                                          min_clip_duration=args.min_duration)
            # raw is expected to be a list of objects with start,end
            highlights = merge_and_normalize_segments(raw, args.max_duration, args.min_duration)
        except Exception as e:
            print("LLM step failed:", e)

    # 4) If still empty, fallback to trivial slicing: make sequential slices up to max_clips
    if not highlights:
        print("Falling back to sequential slicing...")
        # get duration via ffprobe
        cmd = ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", args.input]
        res = subprocess.run(cmd, capture_output=True, text=True)
        total_dur = float(res.stdout.strip())
        clip_len = min(args.max_duration, max(args.min_duration, total_dur / args.max_clips))
        highlights = []
        t = 0.0
        for i in range(args.max_clips):
            if t >= total_dur: break
            end = min(total_dur, t + clip_len)
            highlights.append({"start": t, "end": end, "title": f"Part {i+1}", "reason":"sequential slice", "confidence":0.5})
            t = end

    # 5) Trim clips using ffmpeg
    print(f"Creating {len(highlights)} clips in {args.out_dir}")
    for i, seg in enumerate(tqdm(highlights)):
        out_file = os.path.join(args.out_dir, f"clip_{i+1:02d}.mp4")
        try:
            run_ffmpeg_trim(args.input, out_file, float(seg["start"]), float(seg["end"]), reencode=args.reencode)
        except Exception as e:
            print("Failed to create clip", i+1, e)

    # 6) Save metadata JSON
    meta_file = os.path.join(args.out_dir, "clips_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(highlights, f, indent=2, ensure_ascii=False)
    print("Done. Metadata saved to", meta_file)

if __name__ == "__main__":
    main()
