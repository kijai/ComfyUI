"""Retarget SAM3D Body motion onto another humanoid rig (e.g. a SkinTokens auto-rig) from the MHR70 keypoints."""
import numpy as np
from scipy.spatial.transform import Rotation

from comfy_extras.sam3d_body.export.glb_shared import OPENPOSE_HAND21_TO_MHR70_L, OPENPOSE_HAND21_TO_MHR70_R

NOSE, NECK = 0, 69
# the person's own left/right; fingers are thumb..pinky rows, base to tip
KEYPOINTS = {
    "left": dict(eye=1, ear=3, shoulder=5, elbow=7, hip=9, knee=11, ankle=13, big_toe=15, small_toe=16, heel=17,
                 wrist=62, fingers=OPENPOSE_HAND21_TO_MHR70_L[1:].reshape(5, 4)),
    "right": dict(eye=2, ear=4, shoulder=6, elbow=8, hip=10, knee=12, ankle=14, big_toe=18, small_toe=19, heel=20,
                  wrist=41, fingers=OPENPOSE_HAND21_TO_MHR70_R[1:].reshape(5, 4)),
}
UP = np.array([0.0, 1.0, 0.0])


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def frame(primary, secondary):
    """Orthonormal basis (..., 3, 3): primary, secondary made orthogonal to it, and their cross product."""
    a = unit(primary)
    b = unit(secondary - (secondary * a).sum(-1, keepdims=True) * a)
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def swing(a, b):
    """Shortest-arc rotations (T, 3, 3) taking directions a onto b."""
    a, b = unit(a), unit(b)
    axis = np.cross(a, b)
    angle = np.arctan2(np.linalg.norm(axis, axis=-1), (a * b).sum(-1))
    return Rotation.from_rotvec(unit(axis) * angle[:, None]).as_matrix()


def blend(ra, rb, t):
    """Rotations a fraction t of the way from ra to rb."""
    return Rotation.from_rotvec(Rotation.from_matrix(rb @ ra.transpose(0, 2, 1)).as_rotvec() * t).as_matrix() @ ra


def twist(rel, axis):
    """Angle of rotations rel about unit axes (swing-twist decomposition)."""
    q = Rotation.from_matrix(rel).as_quat()
    q *= np.where(q[:, 3:] < 0, -1.0, 1.0)
    return 2 * np.arctan2((q[:, :3] * axis).sum(-1), q[:, 3])


def match_humanoid(joints, parents):
    """Body roles of a humanoid rig from its tree and Y-up rest positions; facing from the toes, else glTF +Z."""
    children = [[] for _ in parents]
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)

    def lineage(j):
        out = [j]
        while parents[out[-1]] >= 0:
            out.append(parents[out[-1]])
        return out

    def subtree(j):
        out = [j]
        for k in out:
            out.extend(children[k])
        return out

    def run(j):
        out = [j]
        while len(children[out[-1]]) == 1:
            out.append(children[out[-1]][0])
        return out

    def common(a, b):
        mine = lineage(a)
        return next(j for j in lineage(b) if j in mine)

    y = joints[:, 1]
    height = np.ptp(y)
    leaves = sorted((j for j in range(len(parents)) if not children[j]), key=lambda j: y[j])
    # the feet: the two lowest leaves whose paths split well above the floor
    other = next((j for j in leaves[1:] if y[common(leaves[0], j)] > y[leaves[0]] + 0.3 * height), None)
    if other is None:
        raise ValueError("The rig has no pair of legs, retargeting needs a humanoid skeleton.")
    hips = common(leaves[0], other)
    hip_line = lineage(hips)
    legs = [lineage(f)[:lineage(f).index(hips)][::-1] for f in (leaves[0], other)]

    forward = np.array([0.0, 0.0, 1.0])
    toes = [joints[leg[3]] - joints[leg[2]] for leg in legs if len(leg) > 3]
    if toes and np.linalg.norm(np.mean(toes, axis=0) * [1, 0, 1]) > 1e-3 * height:
        forward = unit(np.mean(toes, axis=0) * [1, 0, 1])
    left = np.cross(UP, forward)
    legs.sort(key=lambda leg: joints[leg].mean(axis=0) @ left)

    # the pelvis is where a tall branch leaves the legs' lineage; walk its heaviest children up to the arms
    pelvis = upper = None
    for a in hip_line:
        tall = [c for c in children[a] if c not in hip_line and joints[subtree(c), 1].max() - y[a] > 0.2 * height]
        if tall:
            pelvis, upper = a, max(tall, key=lambda c: len(subtree(c)))
            break
    spine = []
    chest = upper
    while chest is not None:
        sides = ([], [])
        for c in children[chest]:
            sub = subtree(c)
            lateral = (joints[sub].mean(axis=0) - joints[chest]) @ left
            if abs(lateral) > 0.05 * height and len(run(c)) > 1 and np.linalg.norm(joints[sub] - joints[chest], axis=1).max() > 0.15 * height:
                sides[lateral > 0].append((lateral, c))
        if all(sides):
            break
        spine.append(chest)
        chest = max(children[chest], key=lambda c: len(subtree(c)), default=None)
    if chest is None:
        raise ValueError("The rig has no pair of arms, retargeting needs a humanoid skeleton.")
    pair = min(sides[0])[1], max(sides[1])[1]
    arms = []
    for c in pair:
        chain = run(c)
        fingers = [f for f in map(run, children[chain[-1]]) if np.linalg.norm(joints[f] - joints[chain[-1]], axis=1).max() < 0.2 * height]
        if len(fingers) >= 3:
            thumb = fingers.pop(int(np.argmin([np.linalg.norm(joints[f[0]] - joints[chain[-1]]) for f in fingers])))
            across = joints[thumb[0]] - np.mean([joints[f[0]] for f in fingers], axis=0)
            fingers = [thumb] + sorted(fingers, key=lambda f: -(joints[f[0]] @ across))[:4]
        else:
            fingers = []  # a prop or nothing in the hand
        arms.append(dict(segments=chain[:-1], hand=chain[-1], fingers=fingers))

    rest = [c for c in children[chest] if c not in pair]
    neck = run(max(rest, key=lambda c: joints[subtree(c), 1].max())) if rest else []
    return dict(left=left, forward=forward, pelvis=hip_line[:hip_line.index(pelvis) + 1], root=pelvis,
                spine=spine, chest=chest, neck=neck[:1] if len(neck) > 1 else [],
                head=neck[1] if len(neck) > 1 else (neck[0] if neck else None),
                legs=dict(right=legs[0], left=legs[1]), arms=dict(right=arms[0], left=arms[1]))


def retarget(pose_data, rig, person=0):
    """One person of SAM3D Body pose data as frames of `rig` (pose data with a `_skeleton_override` humanoid rig)."""
    o = rig["_skeleton_override"]
    parents = [int(p) for p in o["parents"]]
    bind = np.asarray(o["bind_global_m"], dtype=np.float64)
    rest = bind[:, :3]
    roles = match_humanoid(rest, parents)
    frames = pose_data["frames"]
    present = [t for t, f in enumerate(frames) if person < len(f)]
    if not present:
        raise ValueError(f"The pose data has no person {person}.")
    kp = np.stack([frames[t][person]["pred_keypoints_3d"] for t in present]).astype(np.float64) * [1, -1, -1]
    T = len(present)

    def k(i):
        return kp[:, i]

    def mid(a, b):
        return (a + b) / 2

    L, R = KEYPOINTS["left"], KEYPOINTS["right"]
    legs, arms = roles["legs"], roles["arms"]
    shoulder = {s: arms[s]["segments"][-2] if len(arms[s]["segments"]) > 1 else arms[s]["hand"] for s in arms}

    # full orientations: rest frame of the rig against the same landmarks of the person
    torso = mid(k(L["shoulder"]), k(R["shoulder"])) - mid(k(L["hip"]), k(R["hip"]))
    rig_torso = mid(rest[shoulder["left"]], rest[shoulder["right"]]) - mid(rest[legs["left"][0]], rest[legs["right"][0]])
    fixed = {}
    pelvis = frame(k(L["hip"]) - k(R["hip"]), torso) @ frame(rest[legs["left"][0]] - rest[legs["right"][0]], rig_torso).T
    for j in roles["pelvis"]:
        fixed[j] = pelvis
    fixed[roles["chest"]] = frame(k(L["shoulder"]) - k(R["shoulder"]), torso) @ frame(rest[shoulder["left"]] - rest[shoulder["right"]], rig_torso).T
    if roles["head"] is not None:
        face = (k(NOSE) + k(L["eye"]) + k(R["eye"])) / 3 - mid(k(L["ear"]), k(R["ear"]))
        fixed[roles["head"]] = frame(k(L["ear"]) - k(R["ear"]), face) @ frame(roles["left"], roles["forward"]).T
    swings, blends, forearms = {}, {}, {}
    for side, kps in KEYPOINTS.items():
        leg, arm = legs[side], arms[side]
        fingers = kps["fingers"]
        knuckles = k(fingers[1:, 0]).mean(axis=1)
        if arm["fingers"]:
            rig_along = np.mean([rest[f[0]] for f in arm["fingers"][1:]], axis=0) - rest[arm["hand"]]
            rig_side = rest[arm["fingers"][1][0]] - rest[arm["fingers"][-1][0]]
        else:
            rig_along, rig_side = rest[arm["hand"]] - rest[arm["segments"][-1]], roles["forward"]
        fixed[arm["hand"]] = frame(knuckles - k(kps["wrist"]), k(fingers[1, 0]) - k(fingers[4, 0])) @ frame(rig_along, rig_side).T
        # the other bones point along a keypoint pair
        if len(leg) > 2:
            toes = mid(k(kps["big_toe"]), k(kps["small_toe"]))
            rig_toe = (rest[leg[3]] - rest[leg[2]]) * [1, 0, 1] if len(leg) > 3 else roles["forward"]
            fixed[leg[2]] = frame(toes - k(kps["heel"]), k(kps["ankle"]) - k(kps["heel"])) @ frame(rig_toe, UP).T
            swings[leg[0]] = rest[leg[1]] - rest[leg[0]], k(kps["knee"]) - k(kps["hip"])
            swings[leg[1]] = rest[leg[2]] - rest[leg[1]], k(kps["ankle"]) - k(kps["knee"])
        segments = arm["segments"]
        pairs = [(NECK, kps["shoulder"])] * (len(segments) - 2) + [(kps["shoulder"], kps["elbow"]), (kps["elbow"], kps["wrist"])]
        for j, nxt, (a, b) in zip(segments, segments[1:] + [arm["hand"]], pairs[-len(segments):]):
            swings[j] = rest[nxt] - rest[j], k(b) - k(a)
        if segments:
            forearms[segments[-1]] = arm["hand"]
        for row, finger in zip(fingers, arm["fingers"]):
            for i, j in enumerate(finger[:3]):
                rest_dir = rest[finger[i + 1]] - rest[j] if i + 1 < len(finger) else rest[j] - rest[parents[j]]
                swings[j] = rest_dir, k(row[i + 1]) - k(row[i])
    for chain, a, b in ((roles["spine"], roles["root"], roles["chest"]), (roles["neck"], roles["chest"], roles["head"])):
        for i, j in enumerate(chain):
            blends[j] = a, b, (i + 1) / (len(chain) + 1)

    # world rotations relative to the rest pose, parents first
    order = [j for j, p in enumerate(parents) if p < 0]
    for j in order:
        order.extend(c for c, p in enumerate(parents) if p == j)
    G = {}
    identity = np.broadcast_to(np.eye(3), (T, 3, 3))
    for j in order:
        up = G[parents[j]] if parents[j] >= 0 else identity
        if j in fixed:
            G[j] = fixed[j]
        elif j in blends:
            a, b, t = blends[j]
            G[j] = blend(fixed[a], fixed[b], t)
        elif j in swings:
            rest_dir, target = swings[j]
            G[j] = swing(up @ rest_dir, target) @ up
            if j in forearms:  # half the wrist's twist goes into the forearm
                axis = unit(target)
                G[j] = Rotation.from_rotvec(axis * (twist(fixed[forearms[j]] @ G[j].transpose(0, 2, 1), axis) / 2)[:, None]).as_matrix() @ G[j]
        else:
            G[j] = up

    # positions by forward kinematics, scaled so the rig's legs match the person's stride
    pos = {}
    for j in order:
        p = parents[j]
        pos[j] = np.broadcast_to(rest[j], (T, 3)) if p < 0 else pos[p] + G[p] @ (rest[j] - rest[p])
    rig_leg = np.mean([np.linalg.norm(np.diff(rest[leg[:3]], axis=0), axis=-1).sum() for leg in legs.values()])
    person_leg = np.mean([np.linalg.norm(np.diff(kp[:, [kps["hip"], kps["knee"], kps["ankle"]]], axis=1), axis=-1).sum(-1) for kps in KEYPOINTS.values()], axis=0)
    scale = rig_leg / np.median(person_leg)
    shift = scale * mid(k(L["hip"]), k(R["hip"])) - pos[roles["root"]]
    bind_rot = Rotation.from_quat(bind[:, 3:7]).as_matrix()
    rots = np.stack([G[j] for j in range(len(parents))], axis=1) @ bind_rot
    coords = np.stack([pos[j] for j in range(len(parents))], axis=1) + shift[:, None]
    out = [[] for _ in frames]
    for i, t in enumerate(present):
        out[t] = [{"pred_global_rots": rots[i].astype(np.float32), "pred_joint_coords": coords[i].astype(np.float32),
                   "pred_cam_t": (np.asarray(frames[t][person]["pred_cam_t"]) * scale).astype(np.float32)}]
    return {**rig, "frames": out}
