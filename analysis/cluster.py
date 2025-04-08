#!/usr/bin/env python3

"""Cluster LightDock final swarm results using BSAS algorithm"""

import argparse
from pathlib import Path
from prody import parsePDB, confProDy, calcRMSD
import logging as log
# from lightdock.util.logger import LoggingManager
# from lightdock.constants import CLUSTER_REPRESENTATIVES_FILE
import sys
import numpy as np
from math import sqrt, acos, sin, cos, pi
# Disable ProDy output
confProDy(verbosity="info")
CLUSTER_REPRESENTATIVES_FILE = "cluster.repr"
DEFAULT_ROTATION_STEP = 0.5
# log = LoggingManager.get_logger("lgd_cluster_bsas")
log.basicConfig(
                    format=
                    '[out]-%(levelname)s:%(message)s'
                    )


class Quaternion:

    def __init__(self, w=1., x=0., y=0., z=0.):
        """
        Builds a quaternion.

        If not parameters are defined, returns the identity quaternion
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def clone(self):
        """
        Creates a new instance of this quaternion
        """
        return Quaternion(self.w, self.x, self.y, self.z)

    def __eq__(self, other):
        """
        Compares two quaternions for equality using their components
        """
        return cfloat_equals(self.w, other.w) and \
            cfloat_equals(self.x, other.x) and \
            cfloat_equals(self.y, other.y) and \
            cfloat_equals(self.z, other.z)

    def __ne__(self, other):
        """
        Negation of the __eq__ function
        """
        return not self.__eq__(other)

    def __neg__(self):
        """
        Implements quaternion inverse
        """
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, other):
        """
        Implements quaternion addition
        """
        return Quaternion(self.w+other.w, self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        """
        Implements quaternion substract
        """
        return Quaternion(self.w-other.w, self.x-other.x, self.y-other.y, self.z-other.z)

    def __rmul__(self, scalar):
        """
        Implements multiplication of the form scalar*quaternion
        """
        return Quaternion(scalar * self.w, scalar * self.x, scalar * self.y, scalar * self.z)

    def conjugate(self):
        """
        Calculates the conjugate of this quaternion
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other):
        """
        Calculates quaternion multiplication
        """
        w = (self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z)
        x = (self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y)
        y = (self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x)
        z = (self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w)
        return Quaternion(w, x, y, z)

    def __truediv__(self, scalar):
        """
        Calculates division of quaternion by scalar
        """
        return Quaternion(self.w/scalar, self.x/scalar, self.y/scalar, self.z/scalar)

    def dot(self, other):
        """
        Calculates the dot product of two quaternions
        """
        return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z

    def norm(self):
        """
        Calculates quaternion norm
        """
        return sqrt(self.norm2())

    def norm2(self):
        """
        Calculates quaternion norm^2
        """
        return (self.w*self.w) + (self.x*self.x) + (self.y*self.y) + (self.z*self.z)

    def normalize(self):
        """
        Normalizes a given quaternion
        """
        return self/self.norm()

    def inverse(self):
        """
        Calculates the inverse of this quaternion
        """
        return self.conjugate() / float(self.norm2())

    def rotate(self, vec3):
        """
        Rotates vec3 using quaternion
        """
        v = Quaternion(0., vec3[0], vec3[1], vec3[2])
        r = self*v*self.inverse()
        return [r.x, r.y, r.z]


    def __repr__(self):
        """
        Vector representation of the quaternion
        """
        return "(%10.8f, %10.8f, %10.8f, %10.8f)" % (self.w, self.x, self.y, self.z)

    def lerp(self, other, t):
        """
        Calculates the linear interpolation between two quaternions
        """
        return (1.0-t)*self + t*other

    def slerp(self, other, t=DEFAULT_ROTATION_STEP):
        """
        Calculates the spherical linear interpolation of two quaternions given a t step
        """
        self = self.normalize()
        other = other.normalize()
        q_dot = self.dot(other)
        # Patch to avoid the long path
        if q_dot < 0:
            self = -self
            q_dot *= -1.

        if q_dot > LINEAR_THRESHOLD:
            # Linear interpolation if quaternions are too close
            result = self + t*(other-self)
            result.normalize()
            return result
        else:
            q_dot = max(min(q_dot, 1.0), -1.0)
            omega = acos(q_dot)
            so = sin(omega)
            return (sin((1.0-t)*omega) / so) * self + (sin(t*omega)/so) * other


    def distance(self, other):
        """
        Calculates the closeness of two orientations represented in quaternions space.

        Quaternions must be normalized. Distance is 0 when quaternions are equal and 1 when
        the orientations are 180 degrees apart.
        See http://math.stackexchange.com/questions/90081/quaternion-distance
        """
        return 1-self.dot(other)**2

    @staticmethod
    def random(rng=None):
        """
        Generates a random quaternion uniformly distributed:
        http://planning.cs.uiuc.edu/node198.html
        """
        u1 = 0.
        u2 = 0.
        u3 = 0.
        if rng:
            u1 = rng()
            u2 = rng()
            u3 = rng()
        else:
            u1 = random.random()
            u2 = random.random()
            u3 = random.random()
        return Quaternion(sqrt(1-u1)*sin(2*pi*u2), sqrt(1-u1)*cos(2*pi*u2),
                            sqrt(u1)*sin(2*pi*u3), sqrt(u1)*cos(2*pi*u3))

class DockingResult(object):
    """Represents a LightDock docking result line"""

    def __init__(
        self,
        id_swarm=0,
        id_glowworm=0,
        receptor_id=0,
        ligand_id=0,
        luciferin=0.0,
        num_neighbors=0,
        vision_range=0.0,
        pose=None,
        rmsd=-1.0,
        pdb_file="",
        contacts=0,
        scoring=0.0,
    ):
        self.id_swarm = id_swarm
        self.id_glowworm = id_glowworm
        self.receptor_id = receptor_id
        self.ligand_id = ligand_id
        self.luciferin = luciferin
        self.num_neighbors = num_neighbors
        self.vision_range = vision_range
        self.pose = pose
        self.translation = np.array(pose[:3])
        self.rotation = Quaternion(pose[3], pose[4], pose[5], pose[6]).normalize()
        self.coord = DockingResult.pose_repr(pose)
        self.rmsd = rmsd
        self.pdb_file = pdb_file
        self.contacts = contacts
        self.scoring = scoring

    def __str__(self):
        return "%5d %6d %60s %6d %6d %11.5f %5d %7.3f %8.3f %16s %6d %8.3f" % (
            self.id_swarm,
            self.id_glowworm,
            self.coord,
            self.receptor_id,
            self.ligand_id,
            self.luciferin,
            self.num_neighbors,
            self.vision_range,
            self.rmsd,
            self.pdb_file,
            self.contacts,
            self.scoring,
        )

    def distance_trans(self, other):
        return np.linalg.norm(self.translation - other.translation)

    def distance_rot(self, other):
        return self.rotation.distance(other.rotation)

    @staticmethod
    def pose_repr(coord):
        fields = [("%5.3f" % c) for c in coord]
        return "(%s)" % (", ".join(fields))

def parse_coordinates(line):
    """Parses glowworm's coordinates found in line"""
    first = line.index("(")
    last = line.index(")")
    raw = line[first + 1 : last]
    coord = [float(c) for c in raw.split(",")]
    return coord, first, last

def read_lightdock_output(file_name, initial=None, final=None):
    """Reads a LightDock output file and sorts it by energy"""
    with open(file_name) as fin:
        raw_lines = [line for line in fin if line[0] != "#"]
        results = []
        for id_line, line in enumerate(raw_lines):
            try:
                coord, _, last = parse_coordinates(line)
            except ValueError:
                continue
            rest = line[last + 1 :].split()
            try:
                # Conformer solution
                result = DockingResult(
                    id_glowworm=id_line,
                    receptor_id=int(rest[0]),
                    ligand_id=int(rest[1]),
                    luciferin=float(rest[2]),
                    num_neighbors=int(rest[3]),
                    vision_range=float(rest[4]),
                    pose=coord,
                    scoring=float(rest[5]),
                )
            except ValueError:
                # Default solution
                result = DockingResult(
                    id_glowworm=id_line,
                    receptor_id=0,
                    ligand_id=0,
                    luciferin=float(rest[0]),
                    num_neighbors=int(rest[1]),
                    vision_range=float(rest[2]),
                    pose=coord,
                    scoring=float(rest[3]),
                )
            if initial and final:
                if (id_line + 2) > final:
                    break
                if (id_line + 1) >= initial:
                    results.append(result)
            else:
                results.append(result)

        return results

def parse_command_line():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog="lgd_cluster_bsas")

    parser.add_argument(
        "gso_output_file", help="LightDock output file", metavar="gso_output_file"
    )

    return parser.parse_args()


def get_backbone_atoms(ids_list, swarm_path):
    """Get all backbone atoms (CA or P) of the PDB files specified by the ids_list.

    PDB files follow the format lightdock_ID.pdb where ID is in ids_list
    """
    ca_atoms = {}
    try:
        for struct_id in ids_list:
            pdb_file = swarm_path / f"lightdock_{struct_id}.pdb"
            log.info(f"Reading CA from {pdb_file}")
            structure = parsePDB(str(pdb_file))
            selection = structure.select("name CA P")
            ca_atoms[struct_id] = selection
    except IOError as e:
        log.error(f"Error found reading a structure: {e}")
        log.error(
            "Did you generate the LightDock structures corresponding to this output file?"
        )
        raise SystemExit()
    return ca_atoms


def clusterize(sorted_ids, swarm_path):
    """Clusters the structures identified by the IDS inside sorted_ids list"""

    clusters_found = 0
    clusters = {clusters_found: [sorted_ids[0]]}

    # Read all structures backbone atoms
    backbone_atoms = get_backbone_atoms(sorted_ids, swarm_path)

    for j in sorted_ids[1:]:
        log.info("Glowworm %d with pdb lightdock_%d.pdb" % (j, j))
        in_cluster = False
        for cluster_id in list(clusters.keys()):
            # For each cluster representative
            representative_id = clusters[cluster_id][0]
            rmsd = calcRMSD(backbone_atoms[representative_id], backbone_atoms[j]).round(
                4
            )
            log.info("RMSD between %d and %d is %5.3f" % (representative_id, j, rmsd))
            if rmsd <= 4.0:
                clusters[cluster_id].append(j)
                log.info("Glowworm %d goes into cluster %d" % (j, cluster_id))
                in_cluster = True
                break

        if not in_cluster:
            clusters_found += 1
            clusters[clusters_found] = [j]
            log.info("New cluster %d" % clusters_found)
    return clusters


def write_cluster_info(clusters, gso_data, swarm_path):
    """Writes the clustering result"""
    file_name = swarm_path / CLUSTER_REPRESENTATIVES_FILE
    with open(file_name, "w") as output:
        for id_cluster, ids in clusters.items():
            output.write(
                "%d:%d:%8.5f:%d:%s\n"
                % (
                    id_cluster,
                    len(ids),
                    gso_data[ids[0]].scoring,
                    ids[0],
                    "lightdock_%d.pdb" % ids[0],
                )
            )
        log.info(f"Cluster result written to {file_name} file")


if __name__ == "__main__":

    try:
        # Parse command line
        args = parse_command_line()

        # Read LightDock output data
        gso_data = read_lightdock_output(args.gso_output_file)
        print(args.gso_output_file)
        # Sort the glowworms data by scoring
        sorted_data = sorted(gso_data, key=lambda k: k.scoring, reverse=True)

        # Get the Glowworm ids sorted by their scoring
        sorted_ids = [g.id_glowworm for g in sorted_data]

        # Calculate the different clusters
        swarm_path = Path(args.gso_output_file).absolute().parent
        clusters = clusterize(sorted_ids, swarm_path)

        # Write clustering information
        write_cluster_info(clusters, gso_data, swarm_path)

    except Exception as e:
        log.error("Clustering has failed. Please see error:")
        log.error(str(e))
