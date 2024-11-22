import healpy as hp


rotation_angles = [
        [0, 0, 0],
        [0, 180, 0],
        [180, 180, 0],
        [0, 180, 180],
        [0, 60, 90],
        [0, 120, 90],
        [0, 180, 90],
        [0, 240, 90],
        [0, 300, 90],
        [0, 360, 90],
    ]


def get_rotator(rot_i):
    return hp.Rotator(rot=rotation_angles[rot_i], deg=True, inv=True)
