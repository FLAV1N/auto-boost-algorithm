import statistics
import os
from math import ceil, sqrt
import json
import sys
import subprocess
import copy
from statistics import mean
import vapoursynth as vs

core = vs.core

if "--help" in sys.argv[1:]:
    print('Usage:\npython auto-boost_2.4.py "{animu.mkv}" {base CQ/CRF/Q}"\n\nExample:\npython "auto-boost_2.4.py" "path/to/nice_boat.mkv" 30 6 2 6 0 3 60')
    exit(0)
else:
    pass

og_cq = int(sys.argv[2])  # CQ to start from
WORKERS = int(sys.argv[3])
WORKERS = 6
varboost_strength = int(sys.argv[4])
varboost_octile = int(sys.argv[5])
cdef = int(sys.argv[6])
qp_compress = int(sys.argv[7])
luma_bias = int(sys.argv[8])
cores = ceil(24 / WORKERS)
br = 10  # maximum CQ change from original


# Calculate grain strength for a scene
def calc_grainsynth_of_scene(
    chunk_start,
    chunk_end,
    src,
    encoder_max_grain=50,
) -> int:

    gs = []
    frame_count = chunk_end - chunk_start
    loop_count = int((5 + sqrt(frame_count / 5)) / 2)  # Ensure at least one loop

    for i in range(loop_count):
        curr_frame = chunk_start + int((frame_count / loop_count) * i)
        # Debugging Outputs
        # print(f"Processing frame {curr_frame}...")
        frame_clip = src[curr_frame]
        
        # Apply weak and strong denoising
        weak_clip = core.nlm_cuda.NLMeans(frame_clip, d=0, a=1, s=3, h=1.1, wmode=0)  # Adjust `h` for weak denoising
        
        strong_clip = core.nlm_cuda.NLMeans(frame_clip, d=3, a=3, s=2, h=5.2, wmode=0)  # Adjust `h` for strong denoising 
        
        # Compute sizes for reference, weak, and strong frames
        ref_size = get_size(frame_clip)
        weak_size = get_size(weak_clip)
        strong_size = get_size(strong_clip)

        # Debugging Outputs
        # print(f"Frame {curr_frame}: ref_size={ref_size}, weak_size={weak_size}, strong_size={strong_size}")

        if ref_size == 0 or weak_size == 0 or strong_size == 0:
            print(f"Skipping frame {curr_frame}: Invalid sizes detected.")
            continue

        # Calculate grain factor
        try:
            grain_factor = ref_size * 100.0 / weak_size
            grain_factor = (
                ((ref_size * 100.0 / strong_size * 100.0 / grain_factor) - 105.0)
                * 8.0
                / 10.0
            )
            grain_factor = max(0.0, min(100.0, grain_factor))
            gs.append(grain_factor)
        except Exception as e:
            print(f"Error calculating grain factor for frame {curr_frame}: {e}")
            continue

    if not gs:
        print(f"No grain factors detected for chunk {chunk_start}-{chunk_end}. Defaulting to 1 grain strength.")
        return 1  # Fallback value if no valid grain factor

    final_grain = mean(gs)
    final_grain /= 100.0 / encoder_max_grain
    final_grain = int(round(final_grain))  # Use rounding to avoid truncation
    
    if final_grain <= 3:
        final_grain = 3 

    # Print the final grain value
    # print(f"Chunk {chunk_start}-{chunk_end}: Calculated grain strength {final_grain}")
    return final_grain

def get_size(clip: vs.VideoNode) -> int:
    try:
        common = [
            "ffmpeg",
            "-v", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "-s", f"{clip.width}x{clip.height}",
            "-i", "pipe:",
            "-frames:v", "1",
            "-pix_fmt", "yuv420p",
            "-f", "image2pipe",
            "-c:v", "png", "-"
        ]
        
        process = subprocess.Popen(common, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        clip.output(process.stdin, y4m=False)
        process.stdin.close()
        
        
        total_size = 0

        while process.poll() is None:
            chunk = process.stdout.read(1024 * 1024)
            if chunk:
                total_size += len(chunk)

        process.wait()
        return total_size
    except Exception as e:
        print(f"Error while processing clip with FFmpeg\n{e}")
        return 0

def get_ranges(scenes):
    ranges = []
    ranges.insert(0, 0)
    with open(scenes, "r") as file:
        content = json.load(file)
        for i in range(len(content["scenes"])):
            ranges.append(content["scenes"][i]["end_frame"])
        return ranges


iter = 0
def zones_txt(beginning_frame, end_frame, cq, grain, zones_loc):
    global iter
    iter += 1

    with open(zones_loc, "w" if iter == 1 else "a") as file:
        file.write(f"{beginning_frame} {end_frame} svt-av1 --crf {cq} --film-grain {grain}\n")


def calculate_standard_deviation(score_list: list[int]):
    filtered_score_list = [score for score in score_list if score >= 0]
    sorted_score_list = sorted(filtered_score_list)
    average = sum(filtered_score_list) / len(filtered_score_list)
    return (average, sorted_score_list[len(filtered_score_list) // 20])


fast_av1an_command = f'av1an -i "{sys.argv[1]}" --temp "{sys.argv[1][:-4]}/temp/" -y \
                    --verbose --keep --resume --split-method av-scenechange -m lsmash \
                    --min-scene-len 12 -c mkvmerge --sc-downscale-height 360 \
                    --set-thread-affinity 2 -e svt-av1 --force -v \" \
                    --preset 9 --crf {og_cq} --rc 0 --film-grain 0 --lp 2 \
                    --scm 0 --keyint 0 --fast-decode 1 --enable-tf 0 --enable-cdef {cdef} --variance-boost-strength {varboost_strength} --variance-octile {varboost_octile}  --qp-scale-compress-strength {qp_compress} --chroma-qm-min 10 --frame-luma-bias {luma_bias} --sharpness 2 --color-primaries 1 \
                    --transfer-characteristics 1 --matrix-coefficients 1 \" \
                    --pix-format yuv420p10le -x 240  -w {WORKERS} \
                    -a " -an " -o "{sys.argv[1][:-4]}/{sys.argv[1][:-4]}_fastpass.mkv"'

mypath = f"{sys.argv[1][:-4]}/"
if not os.path.isdir(mypath):
    os.makedirs(mypath)

p = subprocess.Popen(fast_av1an_command, shell=True)
exit_code = p.wait()

if exit_code != 0:
    print("Av1an encountered an error, exiting.")
    exit(-2)

scenes_loc = f"{sys.argv[1][:-4]}/temp/scenes.json"
ranges = get_ranges(scenes_loc)

src = core.lsmas.LWLibavSource(source=sys.argv[1], cache=0)
enc = core.lsmas.LWLibavSource(source=f"{sys.argv[1][:-4]}/{sys.argv[1][:-4]}_fastpass.mkv", cache=0)

print(f"source: {len(src)} frames")
print(f"encode: {len(enc)} frames")

source_clip = src.resize.Bicubic(format=vs.RGBS, matrix_in_s="709").fmtc.transfer(transs="srgb", transd="linear", bits=32)
encoded_clip = enc.resize.Bicubic(format=vs.RGBS, matrix_in_s="709").fmtc.transfer(transs="srgb", transd="linear", bits=32)

percentile_5_total = []
total_ssim_scores: list[int] = []

skip = 10  # amount of skipped frames

for i in range(len(ranges) - 1):
    cut_source_clip = source_clip[ranges[i] : ranges[i + 1]].std.SelectEvery(cycle=skip, offsets=0)
    cut_encoded_clip = encoded_clip[ranges[i] : ranges[i + 1]].std.SelectEvery(cycle=skip, offsets=0)
    result = core.vship.SSIMULACRA2(cut_source_clip, cut_encoded_clip)
    chunk_ssim_scores: list[int] = []

    for index, frame in enumerate(result.frames()):
        score = frame.props["_SSIMULACRA2"]
        # print(f'Frame {index}/{result.num_frames}: {score}')
        chunk_ssim_scores.append(score)
        total_ssim_scores.append(score)

    (average, percentile_5) = calculate_standard_deviation(chunk_ssim_scores)
    percentile_5_total.append(percentile_5)

(average, percentile_5) = calculate_standard_deviation(total_ssim_scores)
print(f"Median score:  {average}\n\n")

for i in range(len(ranges) - 1):
    new_cq = og_cq - ceil((1.0 - (percentile_5_total[i] / average)) * 40 * 4) / 4  # trust me bro

    if new_cq < og_cq - br:  # set lowest allowed cq
        new_cq = og_cq - br

    if new_cq > og_cq + br:  # set highest allowed cq
        new_cq = og_cq + br

    final_grain = calc_grainsynth_of_scene(
        ranges[i], ranges[i + 1], src, encoder_max_grain=50
    )
    print(f"Enc:  [{ranges[i]}:{ranges[i + 1]}]\n"
          f"Chunk 5th percentile: {percentile_5_total[i]}\n"
          f"Adjusted CRF: {new_cq}\n"
          f"Grain: {final_grain}\n\n")
    zones_txt(ranges[i], ranges[i + 1], new_cq, final_grain, f"{scenes_loc[:-11]}zones.txt")