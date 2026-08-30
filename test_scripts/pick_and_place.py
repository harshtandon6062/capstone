"""
Pick and Place Demo — Kuka Robot + WSG50 Gripper
==================================================
Based on the PyBullet pick-and-place tutorial.
Opens a GUI window where the Kuka robot picks up a cube and places it elsewhere.

Run: ~/shared_ml_env/bin/python ~/capstone_project/test_scripts/pick_and_place.py
"""
import time
import math
import numpy as np
import pybullet as p
import pybullet_data

# ──────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Load environment
plane_id = p.loadURDF("plane.urdf")
table_id = p.loadURDF(
    "table/table.urdf",
    basePosition=[1.0, -0.2, 0.0],
    baseOrientation=[0, 0, 0.7071, 0.7071]
)

# Load Kuka robot (VR limits version, positioned on the table edge)
kuka_id = p.loadURDF(
    "kuka_iiwa/model_vr_limits.urdf",
    1.400000, -0.200000, 0.600000,
    0.000000, 0.000000, 0.000000, 1.000000
)

# Load WSG50 gripper
kuka_gripper_id = p.loadSDF(
    "gripper/wsg50_one_motor_gripper_new_free_base.sdf"
)[0]

# Load cube to pick up
cube_id = p.loadURDF(
    "cube.urdf",
    basePosition=[0.85, -0.2, 0.65],
    globalScaling=0.05
)

# ──────────────────────────────────────────────
# ATTACH GRIPPER TO KUKA
# ──────────────────────────────────────────────
kuka_cid = p.createConstraint(
    kuka_id, 6,
    kuka_gripper_id, 0,
    p.JOINT_FIXED,
    [0, 0, 0],
    [0, 0, 0.05],
    [0, 0, 0]
)

# Gear constraint so both gripper fingers move together
kuka_cid2 = p.createConstraint(
    kuka_gripper_id, 4,
    kuka_gripper_id, 6,
    jointType=p.JOINT_GEAR,
    jointAxis=[1, 1, 1],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)
p.changeConstraint(kuka_cid2, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100)

# ──────────────────────────────────────────────
# RESET KUKA TO INITIAL POSE
# ──────────────────────────────────────────────
joint_positions_kuka = [
    -0.000000, -0.000000, 0.000000,
     1.570793,  0.000000, -1.036725,
     0.000001
]
for j_idx in range(p.getNumJoints(kuka_id)):
    p.resetJointState(kuka_id, j_idx, joint_positions_kuka[j_idx])
    p.setJointMotorControl2(kuka_id, j_idx, p.POSITION_CONTROL, joint_positions_kuka[j_idx], 0)

# Reset gripper position and joints
p.resetBasePositionAndOrientation(
    kuka_gripper_id,
    [0.923103, -0.200000, 1.250036],
    [-0.000000, 0.964531, -0.000002, -0.263970]
)
joint_positions_gripper = [
    0.000000, -0.011130, -0.206421, 0.205143,
   -0.009999,  0.000000, -0.010055, 0.000000
]
for j_idx in range(p.getNumJoints(kuka_gripper_id)):
    p.resetJointState(kuka_gripper_id, j_idx, joint_positions_gripper[j_idx])
    p.setJointMotorControl2(kuka_gripper_id, j_idx, p.POSITION_CONTROL, joint_positions_gripper[j_idx], 0)

num_joints = p.getNumJoints(kuka_id)
kuka_end_effector_idx = 6

# ──────────────────────────────────────────────
# SET CAMERA
# ──────────────────────────────────────────────
p.resetDebugVisualizerCamera(
    cameraDistance=2.05,
    cameraYaw=-50,
    cameraPitch=-40,
    cameraTargetPosition=[0.95, -0.2, 0.2]
)

# ──────────────────────────────────────────────
# MAIN PICK AND PLACE LOOP
# ──────────────────────────────────────────────
print("=" * 50)
print("PICK AND PLACE DEMO")
print("=" * 50)
print("Watch the Kuka robot with gripper pick up the cube")
print("and move it to a new position.\n")

total_steps = 750

for t in range(total_steps):
    # ── Determine target position & gripper state ──
    # Default: hover above the cube, gripper open
    target_pos = [0.85, -0.2, 0.97]
    gripper_val = 0  # 0 = open, 1 = closed

    if t < 150:
        # Phase 1: Move above cube (gripper open)
        target_pos = [0.85, -0.2, 0.97]
        gripper_val = 0
        phase = "Moving above cube"

    elif 150 <= t < 250:
        # Phase 2: Close gripper to grab cube
        target_pos = [0.85, -0.2, 0.97]
        gripper_val = 1
        phase = "Gripping cube"

    elif 250 <= t < 400:
        # Phase 3: Lift up with cube
        lift_progress = (t - 250) / 150.0
        target_pos = [0.85, -0.2, 0.97 + 0.13 * lift_progress]
        gripper_val = 1
        phase = "Lifting cube"

    elif 400 <= t < 600:
        # Phase 4: Move sideways to destination
        move_progress = (t - 400) / 200.0
        target_pos = [0.85, -0.2 + 0.4 * move_progress, 1.1]
        gripper_val = 1
        phase = "Moving to destination"

    elif 600 <= t < 700:
        # Phase 5: Hold at destination
        target_pos = [0.85, 0.2, 1.1]
        gripper_val = 1
        phase = "At destination"

    elif t >= 700:
        # Phase 6: Release cube
        target_pos = [0.85, 0.2, 1.1]
        gripper_val = 0
        phase = "Releasing cube"

    # Print phase transitions
    if t in [0, 150, 250, 400, 600, 700]:
        print(f"  Step {t:4d}/{total_steps}: {phase}")

    # ── Apply IK to robot ──
    target_orn = p.getQuaternionFromEuler([0, 1.01 * math.pi, 0])
    joint_poses = p.calculateInverseKinematics(
        kuka_id, kuka_end_effector_idx,
        target_pos, target_orn
    )
    for j in range(num_joints):
        p.setJointMotorControl2(
            bodyIndex=kuka_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_poses[j]
        )

    # ── Control gripper ──
    p.setJointMotorControl2(
        kuka_gripper_id, 4,
        p.POSITION_CONTROL,
        targetPosition=gripper_val * 0.05,
        force=100
    )
    p.setJointMotorControl2(
        kuka_gripper_id, 6,
        p.POSITION_CONTROL,
        targetPosition=gripper_val * 0.05,
        force=100
    )

    p.stepSimulation()
    time.sleep(1 / 120)  # Slow down for visibility

print("\n" + "=" * 50)
print("DEMO COMPLETE!")
print("=" * 50)
print("Close the PyBullet window or press Ctrl+C to exit.\n")

# Keep window open
try:
    while True:
        p.stepSimulation()
        time.sleep(1 / 240)
except KeyboardInterrupt:
    pass

p.disconnect()
