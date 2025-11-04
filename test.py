from rsl_rl.utils.motion_loader import Motion


motion_loader = Motion()
motion_loader.load_motions(
    motion_folder="./AMP_Motion",
    motion_files=["AMPdebug"],
    weights=[1.0],
    target_fps=30,
)
print(motion_loader.action_pairs[0])