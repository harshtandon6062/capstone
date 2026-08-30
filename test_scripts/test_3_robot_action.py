"""
TEST 3: Robot Performing Action (Pick & Place)
===============================================
Tests the Kuka robot arm doing IK-based pick-and-place in PyBullet.
No webcam needed — runs automatically.

Run: ~/shared_ml_env/bin/python test_scripts/test_3_robot_action.py
"""
import pybullet as p
import pybullet_data
import time
import numpy as np

print("=" * 50)
print("ROBOT ACTION TEST")
print("=" * 50)

# Setup simulation
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")
table = p.loadURDF("table/table.urdf", [0.5, 0, 0])

# Robot base on the FLOOR, offset from the table so it doesn't clip through
robot = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0.0, 0, 0],
    useFixedBase=True
)

# Set a nice camera angle to watch the demo
p.resetDebugVisualizerCamera(
    cameraDistance=1.8,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0.4, 0, 0.5]
)

# Spawn cubes on the table surface (~z=0.65)
objects = []
for i in range(3):
    obj = p.loadURDF(
        "cube_small.urdf",
        basePosition=[0.5, -0.1 + i * 0.1, 0.66]
    )
    objects.append(obj)

print(f"Loaded {len(objects)} objects on table")
print(f"Robot has {p.getNumJoints(robot)} joints")

# Let objects settle
print("\nLetting objects settle (2 seconds)...")
for _ in range(480):
    p.stepSimulation()
    time.sleep(1/240)
time.sleep(1)


def move_robot(robot_id, target_pos, steps=300):
    """Move end-effector to target position using IK (SLOW for visibility)"""
    joint_positions = p.calculateInverseKinematics(robot_id, 6, target_pos)
    for j in range(len(joint_positions)):
        p.setJointMotorControl2(
            robot_id, j,
            p.POSITION_CONTROL,
            joint_positions[j],
            force=500
        )
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1/120)  # Slower than real-time so you can watch


# Demo: move robot to each object position
print("\n--- Moving robot to each object (watch the 3D window!) ---")

for i, obj in enumerate(objects):
    pos = p.getBasePositionAndOrientation(obj)[0]
    above_pos = [pos[0], pos[1], pos[2] + 0.2]

    print(f"  >> Moving to object {i+1}...")
    move_robot(robot, above_pos)
    print(f"     Arrived at object {i+1}. Pausing 2s...")
    time.sleep(2)

# Return to home position
print("\n  >> Returning to home position...")
move_robot(robot, [0.3, 0, 0.9])
time.sleep(2)

# Demo: sweep motion (move across the table)
print("\n--- Sweep motion demo (watch robot slide across table) ---")
time.sleep(1)
for i, x in enumerate(np.linspace(0.3, 0.6, 8)):
    move_robot(robot, [x, 0, 0.75], steps=100)
    print(f"  Sweep step {i+1}/8")

print("\nRobot action test complete!")
print("Close the PyBullet window or press Ctrl+C to exit.")

# Keep simulation running so user can see
try:
    while True:
        p.stepSimulation()
        time.sleep(1/240)
except KeyboardInterrupt:
    pass

p.disconnect()
