# SkinTokens (MIT) skeleton token grammar and detokenizer: https://github.com/VAST-AI-Research/SkinTokens
import numpy as np

NUM_DISCRETE = 256
BRANCH = 256
BOS = 257
EOS = 258  # ends the skeleton; skin tokens follow
PAD = 259
SPRING = 260
PARTS = (261, 262)  # body, hand
CLS_NONE = 263
CLS = {"rignet": 264, "vroid": 265, "articulation": 266}
VOCAB_SIZE = 267
CONTINUOUS_RANGE = (-1.0, 1.0)

# Grammar states of TokenizerPart.next_posible_token
EXPECT_BOS, EXPECT_CLS_OR_PART_OR_JOINT, EXPECT_PART_OR_JOINT, EXPECT_JOINT_2, EXPECT_JOINT_3, EXPECT_BRANCH_OR_PART_OR_JOINT, EXPECT_JOINT = range(7)


def allowed_tokens(state):
    joints = list(range(NUM_DISCRETE))
    parts = [SPRING, *PARTS]
    if state == EXPECT_BOS:
        return [BOS]
    if state == EXPECT_CLS_OR_PART_OR_JOINT:
        return [CLS_NONE, *CLS.values()] + parts + joints
    if state == EXPECT_PART_OR_JOINT:
        return parts + joints + [EOS]
    if state == EXPECT_BRANCH_OR_PART_OR_JOINT:
        return joints + parts + [BRANCH, EOS]
    return joints


class SkeletonGrammar:
    """Incremental equivalent of TokenizerPart.next_posible_token / bones_in_sequence."""

    def __init__(self):
        self.state = EXPECT_BOS
        self.num_bones = 0
        self.is_branch = False
        self.skin_tokens = None  # tokens after EOS, None while still in the skeleton

    def copy(self):
        g = SkeletonGrammar()
        g.state, g.num_bones, g.is_branch, g.skin_tokens = self.state, self.num_bones, self.is_branch, self.skin_tokens
        return g

    def push(self, token):
        if self.skin_tokens is not None:
            self.skin_tokens += 1
            return
        if token == EOS:
            self.skin_tokens = 0
            return
        s = self.state
        if s == EXPECT_BOS:
            self.state = EXPECT_CLS_OR_PART_OR_JOINT
        elif s == EXPECT_CLS_OR_PART_OR_JOINT:
            if token < NUM_DISCRETE:
                self.state = EXPECT_JOINT_2
            elif token == CLS_NONE or token in CLS.values():
                self.state = EXPECT_PART_OR_JOINT
            else:
                self.state = EXPECT_JOINT
        elif s == EXPECT_PART_OR_JOINT:
            if token < NUM_DISCRETE:
                self.state = EXPECT_JOINT_2
        elif s == EXPECT_JOINT_2:
            self.state = EXPECT_JOINT_3
        elif s == EXPECT_JOINT_3:
            if not self.is_branch:
                self.num_bones += 1
            self.is_branch = False
            self.state = EXPECT_BRANCH_OR_PART_OR_JOINT
        elif s == EXPECT_BRANCH_OR_PART_OR_JOINT:
            if token == BRANCH:
                self.is_branch = True
                self.state = EXPECT_JOINT
            elif token < NUM_DISCRETE:
                self.state = EXPECT_JOINT_2
            else:
                self.state = EXPECT_JOINT
        else:
            self.state = EXPECT_JOINT_2


def undiscretize(t):
    lo, hi = CONTINUOUS_RANGE
    return (t.astype(np.float32) + 0.5) / NUM_DISCRETE * (hi - lo) + lo


def detokenize(ids):
    """Skeleton tokens (BOS ... EOS) -> joints (J, 3) and parents (J,)."""
    ids = ids[1:-1]
    joints = []
    p_joints = []
    i = 0
    is_branch = False
    last_joint = None
    while i < len(ids):
        if ids[i] < NUM_DISCRETE:
            if is_branch:
                if i + 6 > len(ids):  # the grammar allows EOS after a branch's parent, which adds no joint
                    break
                p_joint = undiscretize(ids[i:i + 3])
                current_joint = undiscretize(ids[i + 3:i + 6])
                i += 6
            else:
                current_joint = undiscretize(ids[i:i + 3])
                p_joint = current_joint if len(p_joints) == 0 else last_joint
                i += 3
            joints.append(current_joint)
            p_joints.append(p_joint)
            last_joint = current_joint
            is_branch = False
        elif ids[i] == BRANCH:
            is_branch = True
            last_joint = None
            i += 1
        else:  # cls / part tokens only name joints
            i += 1

    # make_skeleton: the parent is the latest earlier joint closest to the encoded parent position
    parents = [-1]
    for i in range(1, len(joints)):
        dis = 999999
        pid = None
        for j in reversed(range(i)):
            n_dis = ((joints[j] - p_joints[i]) ** 2).sum()
            if n_dis < dis:
                pid = j
                dis = n_dis
        parents.append(pid)
    return np.stack(joints), parents
