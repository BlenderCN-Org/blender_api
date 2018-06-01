# Implements the commands defined by the public API
import bpy
from mathutils import Matrix, Euler
from math import pi
from collections import OrderedDict
import logging
import math
from rigAPI.rigAPI import RigAPI

logger = logging.getLogger('hr.blender_api.rigcontrol.commands')
# ====================================================

def init():
    bpy.ops.wm.animation_playback()
    return 0

def getEnvironment():
    return None

def terminate():
    return 0


def rad2deg(x):
    return (x * 180.0) / 3.1415927

def clamp(x,min,max):
    result = x
    if result < min:
        result = min
    if result > max:
        result = max
    return result

def get_local_matrix(b):
    rest = b.bone.matrix_local.copy()
    rest_inv = rest.inverted()
    if b.parent:
        par_mat = b.parent.matrix.copy()
        par_inv = par_mat.inverted()
        par_rest = b.parent.bone.matrix_local.copy()
    else:
        par_mat = Matrix()
        par_inv = Matrix()
        par_rest = Matrix()
    return rest_inv * (par_rest * (par_inv * b.matrix))

class EvaAPI(RigAPI):
    PAU_HEAD_YAW = 1
    PAU_HEAD_PITCH = 2
    PAU_HEAD_ROLL = 4
    PAU_EYE_TARGET = 8
    PAU_FACE = 16
    PAU_ARMS = 32
    # Flag which determines if currenjtly PAU messages are being transmitted
    PAU_ACTIVE = 128
    PAU_ACTIVE_TIMEOUT = 0.5

    ARM_ROTATIONS = {
        'R_Shoulder_Pitch'  : 'Shoulder_R:0',
        'R_Shoulder_Roll'   : 'Shoulder_R:2',
        'R_Shoulder_Yaw'    : 'Arm_Twist_R:1',
        'R_Elbow'           : 'Elbow_R:0',
        'R_Wrist_Yaw'       : 'Forearm_Twist_R:1',

        'R_Wrist_Roll'      : 'Wrist_R:2',
        'R_Index_Finger'    : 'Index_Fing_Base_R:0',
        'R_Middle_Finger'   : 'Mid_Base_R:0',
        'R_Ring_Finger'     : 'Ring_Base_R:0',
        'R_Pinky_Finger'    : 'Pinky_Base_R:0',
        'R_Thumb_Finger'    : 'Thumb_Base_R:0',
        'R_Thumb_Roll'      : 'Thumb_Pivot_R:1',
        'R_Spreading'       : 'Thumb_Pivot_R:0',

        'L_Shoulder_Pitch'  : 'Shoulder_L:0',
        'L_Shoulder_Roll'   : 'Shoulder_L:2',
        'L_Shoulder_Yaw'    : 'Arm_Twist_L:1',
        'L_Elbow'           : 'Elbow_L:0',
        'L_Wrist_Yaw'       : 'Forearm_Twist_L:1',

        'L_Wrist_Roll'      : 'Wrist_L:2',
        'L_Index_Finger'    : 'Index_Fing_Base_L:0',
        'L_Middle_Finger'   : 'Mid_Base_L:0',
        'L_Ring_Finger'     : 'Ring_Base_L:0',
        'L_Pinky_Finger'    : 'Pinky_Base_L:0',
        'L_Thumb_Finger'    : 'Thumb_Base_L:0',
        'L_Thumb_Roll'      : 'Thumb_Pivot_L:1',
        'L_Spreading'       : 'Thumb_Pivot_L:0',
    }

    def __init__(self):
        self.armsAnimationMode = 0 # start sitting
        # Current animation mode (combined by addition)
        # 0 - Face eyes and head controlled by animations
        # 1 - head yaw controlled by PAU
        # 2 - head pitch controlled by PAU
        # 4 - head roll controlled by PAU
        # 8 - Eye Target controlled by PAU
        # 16 - Face shapekeys controlled by PAU
        # 32 - Arms controlled by PAU
        self.pauAnimationMode = 0
        # If 1 current shapekeys are controlled directly by PAU, otherwise by default drivers
        self.shapekeysControl = 0
        # Time for PAU controls to expire
        self.pauTimeout = 0
        pass


    def getAPIVersion(self):
        return 4

    def isAlive(self):
        return int(bpy.data['animationPlaybackActive'])

    def setArmsMode(self,arms_animation_mode):
        self.armsAnimationMode = arms_animation_mode
        bpy.evaAnimationManager.setArmsMode(self.armsAnimationMode)
        return True

    def getArmsMode(self):
        return self.armsAnimationMode


    # Faceshift to ROS mapping functions
    def getAnimationMode(self):

        return self.pauAnimationMode

    def setAnimationMode(self, animation_mode):

        ## Now let's delete the shape
        if self.pauAnimationMode != animation_mode:
            print(animation_mode)
            # Face should drivers should be disabled
            # Face drivers are enabled on the first PAU message recieved if the correct animation mode is set.
            if animation_mode & (self.PAU_FACE | self.PAU_ACTIVE) == (self.PAU_FACE | self.PAU_ACTIVE):
                bpy.evaAnimationManager.setMode(1)
            else:
                 bpy.evaAnimationManager.setMode(0)
            self.pauAnimationMode = animation_mode
        return 0

    def setShapeKeys(self, shape_keys):
        bpy.evaAnimationManager.setShapeKeys(shape_keys)
        return 0
    # Somatic states  --------------------------------
    # awake, asleep, drunk, dazed and confused ...
    def availableSomaStates(self):
        somaStates = []
        for state in bpy.data.actions:
            if state.name.startswith("CYC-"):
                somaStates.append(state.name[4:])
        return somaStates

    def getSomaStates(self):
        eva = bpy.evaAnimationManager
        somaStates = {}
        for cycle in eva.cyclesSet:
            magnitude = round(cycle.magnitude, 3)
            rate = round(cycle.rate, 3)
            ease_in = round(cycle.ease_in, 3)
            somaStates[cycle.name] = {'magnitude': magnitude, 'rate': rate,
                'ease_in': ease_in}
        return somaStates

    def setSomaState(self, state):
        name = 'CYC-' + state['name']
        rate = state['rate']
        magnitude = state['magnitude']
        ease_in = state['ease_in']
        bpy.evaAnimationManager.setCycle(name=name,
            rate=rate, magnitude=magnitude, ease_in=ease_in)
        return 0

    # Emotion expressions ----------------------------
    # smiling, frowning, bored ...
    def availableEmotionStates(self):
        emotionStates = []
        for emo in bpy.data.objects['control'].pose.bones:
            if emo.name.startswith('EMO-'):
                emotionStates.append(emo.name[4:])
        return emotionStates


    def getEmotionStates(self):
        eva = bpy.evaAnimationManager
        emotionStates = {}
        for emotion in eva.emotionsList:
            magnitude = round(emotion.magnitude.current, 3)
            emotionStates[emotion.name] = {'magnitude': magnitude}
        return emotionStates


    def setEmotionState(self, emotion):
        # TODO: expand arguments and update doc
        bpy.evaAnimationManager.setEmotion(eval(emotion))
        return 0

    def setEmotionValue(self, emotion):
        # TODO: expand arguments and update doc
        bpy.evaAnimationManager.setEmotionValue(eval(emotion))
        return 0


    # Gestures --------------------------------------
    # blinking, nodding, shaking...
    def availableGestures(self):
        emotionGestures = []
        for gesture in bpy.data.actions:
            if gesture.name.startswith("GST-"):
                emotionGestures.append(gesture.name[4:])
        return emotionGestures


    def getGestures(self):
        eva = bpy.evaAnimationManager
        emotionGestures = {}
        for gesture in eva.gesturesList:
            duration = round(gesture.duration*gesture.repeat - gesture.stripRef.strip_time, 3)
            magnitude = round(gesture.magnitude, 3)
            speed = round(gesture.speed, 3)
            emotionGestures[gesture.name] = {'duration': duration, \
                'magnitude': magnitude, 'speed': speed}
        return emotionGestures


    def setGesture(self, name, repeat=1, speed=1, magnitude=1.0):
        bpy.evaAnimationManager.newGesture(name='GST-'+name, \
            repeat=repeat, speed=speed, magnitude=magnitude)
        return 0


    def stopGesture(self, gestureID, smoothing):
        ## TODO
        return 0

    # Arm animations --------------------------------------
    def availableArmAnimations(self):
        armAnimations = []
        for armanimation in bpy.data.actions:
            if armanimation.name.startswith("ARM-"):
                armAnimations.append(armanimation.name[4:])
        return armAnimations


    def getArmAnimations(self):
        eva = bpy.evaAnimationManager
        armAnimations = {}
        for armanimation in eva.armAnimationsList:
            duration = round(armanimation.duration*armanimation.repeat - armanimation.stripRef.strip_time, 3)
            magnitude = round(armanimation.magnitude, 3)
            speed = round(armanimation.speed, 3)
            armAnimations[armanimation.name] = {'duration': duration, \
                'magnitude': magnitude, 'speed': speed}
        return armAnimations


    def setArmAnimation(self, name, repeat=1, speed=1, magnitude=1.0):
        bpy.evaAnimationManager.newArmAnimation(name='ARM-'+name, \
            repeat=repeat, speed=speed, magnitude=magnitude)
        return 0


    def stopArmAnimation(self, gestureID, smoothing):
        ## TODO
        return 0

    # Visemes --------------------------------------
    def availableVisemes(self):
        visemes = []
        for viseme in bpy.data.actions:
            if viseme.name.startswith("VIS-"):
                visemes.append(viseme.name[4:])
        return visemes


    def queueViseme(self, vis, start=0, duration=0.5, \
            rampin=0.1, rampout=0.8, magnitude=1):
        return bpy.evaAnimationManager.newViseme("VIS-"+vis, duration, \
            rampin, rampout, start)

    # Eye look-at targets ==========================
    # The coordinate system used is head-relative, in 'engineering'public_ws/src/blender_api/rigControl/commands.py:135
    # coordinates: 'x' is forward, 'y' to the left, and 'z' up.
    # Distances are measured in meters.  Origin of the coordinate
    # system is somewhere (where?) in the middle of the head.

    def setFaceTarget(self, loc, speed=1.0):
        # Eva uses y==forward x==right. Distances in meters from
        # somewhere in the middle of the head.
        mloc = [loc[1], loc[0], loc[2]]
        bpy.evaAnimationManager.setFaceTarget(mloc, speed)
        return 0

    # Rotates the face target which will make head roll
    def setHeadRotation(self,rot):
        bpy.evaAnimationManager.setHeadRotation(rot)
        return 0


    def setGazeTarget(self, loc, speed=1.0):
        mloc = [loc[1],  loc[0], loc[2]]
        bpy.evaAnimationManager.setGazeTarget(mloc, speed)
        return 0
    # ========== procedural animations with unique parameters =============
    def setBlinkRandomly(self,interval_mean,interval_variation):
        bpy.evaAnimationManager.setBlinkRandomly(interval_mean,interval_variation)
        return 0

    def setSaccade(self,interval_mean,interval_variation,paint_scale,eye_size,eye_distance,mouth_width,mouth_height,weight_eyes,weight_mouth):
        bpy.evaAnimationManager.setSaccade(interval_mean,interval_variation,paint_scale,eye_size,eye_distance,mouth_width,mouth_height,weight_eyes,weight_mouth)
        return 0

    # ========== info dump for ROS, Should return non-blender data structures

    # Gets Head rotation quaternion in XYZ format in blender independamt
    # data structure.
    # Pitch: X (positive down, negative up)?
    # Yaw: Z (negative right to positive left)
    #
    # The bones['DEF-head'].id_data.matrix_world currently return the
    # unit matrix, and so are not really needed.

    def getHeadData(self):
        bones = bpy.evaAnimationManager.deformObj.pose.bones
        rhead = bones['DEF-head'].matrix * Matrix.Rotation(-pi/2, 4, 'X')
        rneck = bones['DEF-neck'].matrix * Matrix.Rotation(-pi/2, 4, 'X')
        rneck.invert()

        # I think this is the correct order for the neck rotations.
        q = (rneck * rhead).to_quaternion()
        # q = (rhead * rneck).to_quaternion()
        return {'x':q.x, 'y':q.y, 'z':q.z, 'w':q.w}

    # Same as head, but for the lower neck joint.
    def getNeckData(self):
        bones = bpy.evaAnimationManager.deformObj.pose.bones
        rneck = bones['DEF-neck'].matrix * Matrix.Rotation(-pi/2, 4, 'X')
        q = rneck.to_quaternion()
        return {'x':q.x, 'y':q.y, 'z':q.z, 'w':q.w}

    # Gets Eye rotation angles:
    # Pitch: down(negative) to up(positive)
    # Yaw: left (negative) to right(positive)

    def getEyesData(self):
        bones = bpy.evaAnimationManager.deformObj.pose.bones
        head = (bones['DEF-head'].id_data.matrix_world*bones['DEF-head'].matrix*Matrix.Rotation(-pi/2, 4, 'X')).to_euler()
        leye = bones['eye.L'].matrix.to_euler()
        reye = bones['eye.R'].matrix.to_euler()
        # Relative to head. Head angles are inversed.
        leye_p = leye.x + head.x
        leye_y = pi - leye.z if leye.z >= 0 else -(pi+leye.z)
        reye_p = reye.x + head.x
        reye_y = pi - reye.z if reye.z >= 0 else -(pi+reye.z)
        # Add head target
        leye_y += head.z
        reye_y += head.z
        return {'l':{'p':leye_p,'y':leye_y},'r':{'p':reye_p,'y':reye_y}}


    def getFaceData(self):
        shapekeys = OrderedDict()
        for shapekeyGroup in bpy.data.shape_keys:
            # Hardcoded to find the correct group
            if shapekeyGroup.name == 'ShapeKeys':
                for kb in shapekeyGroup.key_blocks:
                    shapekeys[kb.name] = kb.value

        # Fake the jaw shapekey from its z coordinate
        jawz = bpy.evaAnimationManager.deformObj.pose.bones['chin'].location[2]
        shapekeys['jaw'] = min(max(jawz*7.142, 0), 1)

        return shapekeys

    def get_arm_joints(self, joint):
        (rot,key) = self.ARM_ROTATIONS[joint].split(':')
        bones = bpy.evaAnimationManager.skeleton.pose.bones
        return bones[rot].rotation_euler[int(key)]

    def set_arm_joint(self, joint, angle):
        (rot,key) = self.ARM_ROTATIONS[joint].split(':')
        bones = bpy.evaAnimationManager.skeleton.pose.bones
        bones[rot].rotation_euler[int(key)] = math.radians(angle)

    def getArmsData(self):
        if not bpy.evaAnimationManager.arms_enabled:
            return {}
        angles = OrderedDict()
        angles['R_Shoulder_Pitch'] = rad2deg(self.get_arm_joints('R_Shoulder_Roll'))
        angles['R_Shoulder_Roll'] = rad2deg(self.get_arm_joints('R_Shoulder_Roll'))
        angles['R_Shoulder_Yaw'] = rad2deg(self.get_arm_joints('R_Shoulder_Yaw'))
        angles['R_Elbow'] = rad2deg(self.get_arm_joints('R_Elbow'))
        angles['R_Wrist_Yaw'] = rad2deg(self.get_arm_joints('R_Wrist_Yaw'))

        angles['R_Wrist_Roll'] = rad2deg(self.get_arm_joints('R_Wrist_Roll'))
        angles['R_Index_Finger'] = rad2deg(self.get_arm_joints('R_Index_Finger'))
        angles['R_Middle_Finger'] = rad2deg(self.get_arm_joints('R_Middle_Finger'))
        angles['R_Ring_Finger'] = rad2deg(self.get_arm_joints('R_Ring_Finger'))
        angles['R_Pinky_Finger'] = rad2deg(self.get_arm_joints('R_Pinky_Finger'))
        angles['R_Thumb_Finger'] = rad2deg(self.get_arm_joints('R_Thumb_Finger'))
        angles['R_Thumb_Roll'] = rad2deg(self.get_arm_joints('R_Thumb_Roll'))
        angles['R_Spreading'] = rad2deg(self.get_arm_joints('R_Spreading'))

        angles['L_Shoulder_Pitch'] = rad2deg(self.get_arm_joints('L_Shoulder_Pitch'))
        angles['L_Shoulder_Roll'] = rad2deg(self.get_arm_joints('L_Shoulder_Roll'))
        angles['L_Shoulder_Yaw'] = rad2deg(self.get_arm_joints('L_Shoulder_Yaw'))
        angles['L_Elbow'] = rad2deg(self.get_arm_joints('L_Elbow'))
        angles['L_Wrist_Yaw'] = rad2deg(self.get_arm_joints('L_Wrist_Yaw'))

        angles['L_Wrist_Roll'] = rad2deg(self.get_arm_joints('L_Wrist_Roll'))
        angles['L_Index_Finger'] = rad2deg(self.get_arm_joints('L_Index_Finger'))
        angles['L_Middle_Finger'] = rad2deg(self.get_arm_joints('L_Middle_Finger'))
        angles['L_Ring_Finger'] = rad2deg(self.get_arm_joints('L_Ring_Finger'))
        angles['L_Pinky_Finger'] = rad2deg(self.get_arm_joints('L_Pinky_Finger'))
        angles['L_Thumb_Finger'] = rad2deg(self.get_arm_joints('L_Thumb_Finger'))
        angles['L_Thumb_Roll'] = rad2deg(self.get_arm_joints('L_Thumb_Roll'))
        angles['L_Spreading'] = rad2deg(self.get_arm_joints('L_Spreading'))

        # Exceptions for sitting mode
        if bpy.evaAnimationManager.armsAnimationMode == 0:  # sitting (safe)
            angles['R_Shoulder_Pitch'] = clamp(-20.0 + 70.0 * (rad2deg(self.get_arm_joints('R_Shoulder_Pitch')) / 90.0),
                                               -90.0, -20.0)
            angles['R_Shoulder_Roll'] = clamp(45.0 * (rad2deg(self.get_arm_joints('R_Shoulder_Roll')) / 90.0), 0.0, 45.0)
            angles['R_Shoulder_Yaw'] = clamp(-10.0 + 25.0 * (rad2deg(self.get_arm_joints('R_Shoulder_Yaw')) / 45.0),
                                             -35.0, 15.0)
            angles['R_Elbow'] = clamp(50.0 + 60.0 * (rad2deg(self.get_arm_joints('R_Elbow')) / 90.0),
                                      50.0, 110.0)

            angles['L_Shoulder_Pitch'] = clamp(-20.0 + 70.0 * (rad2deg(self.get_arm_joints('L_Shoulder_Pitch')) / 90.0), -90.0,
                                               -20.0)
            angles['L_Shoulder_Roll'] = clamp(45.0 * (rad2deg(self.get_arm_joints('L_Shoulder_Roll')) / 90.0), -45.0, 0.0)
            angles['L_Shoulder_Yaw'] = clamp(10.0 + 25.0 * (rad2deg(self.get_arm_joints('L_Shoulder_Yaw')) / 45.0), -15.0,
                                             35.0)
            angles['L_Elbow'] = clamp(50.0 + 60.0 * (rad2deg(self.get_arm_joints('L_Elbow')) / 90.0), 50.0, 110.0)

        return angles

    def setArmsJoints(self, joints):
        for k,v in joints.items():
            self.set_arm_joint(k,v)

    def setNeckRotation(self, pitch, roll):
        bpy.evaAnimationManager.deformObj.pose.bones['DEF-neck'].rotation_euler = Euler((pitch, 0, roll))

    def setParam(self, key, value):
        cmd = "%s=%s" % (str(key), str(value))
        logger.info("Run %s" % cmd)
        try:
            exec(cmd)
        except Exception as ex:
            logger.error("Error %s" % ex)
            return False
        return True

    def getParam(self, param):
        param = param.strip()
        logger.info("Get %s" % param)
        try:
            return str(eval(param))
        except Exception as ex:
            logger.error("Error %s" % ex)

    def getAnimationLength(self, animation):
        animation = "GST-"+animation
        if not animation in bpy.data.actions.keys():
            return 0
        else:
            frame_range = bpy.data.actions[animation].frame_range
            frames = 1+frame_range[1]-frame_range[0]
            return frames / bpy.context.scene.render.fps

    def getArmAnimationLength(self, animation):
        animation = "ARM-"+animation
        if not animation in bpy.data.actions.keys():
            return 0
        else:
            frame_range = bpy.data.actions[animation].frame_range
            frames = 1+frame_range[1]-frame_range[0]
            return (frames * 2) / bpy.context.scene.render.fps

    def getCurrentFrame(self):
        if bpy.context.object.animation_data.action is not None:
            name = bpy.context.object.animation_data.action.name
            frame = bpy.context.scene.frame_current
            return (name, frame)
