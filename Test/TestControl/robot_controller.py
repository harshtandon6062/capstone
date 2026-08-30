import pybullet as p
import time

def move_robot(robot,target_pos):

    joint_positions=p.calculateInverseKinematics(
        robot,
        6,
        target_pos
    )

    for j in range(len(joint_positions)):

        p.setJointMotorControl2(
            robot,
            j,
            p.POSITION_CONTROL,
            joint_positions[j],
            force=500
        )

    for _ in range(120):
        p.stepSimulation()
        time.sleep(1/240)

def execute_pick_place(robot,source_obj,dest_obj):

    src=p.getBasePositionAndOrientation(source_obj)[0]
    dst=p.getBasePositionAndOrientation(dest_obj)[0]

    above_src=[src[0],src[1],src[2]+0.2]
    above_dst=[dst[0],dst[1],dst[2]+0.2]

    move_robot(robot,above_src)
    move_robot(robot,above_dst)