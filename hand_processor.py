import numpy as np

class HandProcessor:
    """
    Transforms raw MediaPipe landmarks into normalized, 
    position-independent features suitable for gesture recognition.
    """

    def normalize(self, landmarks, img_shape):
        """
        Normalize landmarks relative to the wrist (point 0).
        
        Why: Raw x/y are screen positions. If you move your hand 
        left, every coordinate changes — but the gesture is the same.
        By subtracting the wrist and scaling by hand size, we get 
        a description of SHAPE only, not position.
        
        Returns: list of 21 dicts with normalized x, y, z
        """
        h, w, _ = img_shape
        # Convert to pixel coordinates
        pts = [(lm.x * w, lm.y * h, lm.z) for lm in landmarks]
        
        # Translate so wrist (pt 0) is the origin
        wx, wy, wz = pts[0]
        pts = [(x - wx, y - wy, z - wz) for x, y, z in pts]
        
        # Scale by the distance from wrist to middle finger base (pt 9)
        # This makes the gesture size-independent too
        scale = np.sqrt(pts[9][0]**2 + pts[9][1]**2) or 1
        pts = [(x/scale, y/scale, z/scale) for x, y, z in pts]
        
        return [{"x": x, "y": y, "z": z} for x, y, z in pts]

    def extract_angles(self, normalized_pts):
        """
        Compute the bend angle at each finger joint.
        
        Why: Angles are even more robust than coordinates — they're 
        invariant to rotation as well as position. A fist looks like 
        a fist from any direction when described by joint angles.
        
        Returns: list of 15 angles (3 joints × 5 fingers)
        """
        # Finger joint index triplets: (base, middle, tip) for angle calc
        finger_joints = [
            [1, 2, 3],   # Thumb joints
            [5, 6, 7],   # Index
            [9, 10, 11], # Middle
            [13, 14, 15],# Ring
            [17, 18, 19] # Pinky
        ]
        angles = []
        pts = [(p["x"], p["y"], p["z"]) for p in normalized_pts]

        for joint_group in finger_joints:
            for i in range(len(joint_group) - 1):
                a = np.array(pts[joint_group[i]])
                b = np.array(pts[joint_group[i+1]])
                # Vector from a to b
                v = b - a
                # Angle relative to downward axis (simplified)
                angle = np.degrees(np.arctan2(v[1], v[0]))
                angles.append(round(angle, 2))
        return angles

    def to_flat_vector(self, normalized_pts):
        """
        Flatten normalized landmarks to a 1D list of 63 numbers (21 × 3).
        Used as input to classifiers.
        """
        return [val for pt in normalized_pts for val in (pt["x"], pt["y"], pt["z"])]