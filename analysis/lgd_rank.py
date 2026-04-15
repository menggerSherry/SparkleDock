#!/usr/bin/env python3

"""Calculates the ranking files depending of different metrics"""

import os
import argparse
import operator
# from lightdock.constants import (
#     DEFAULT_SWARM_FOLDER,
#     GSO_OUTPUT_FILE,
#     EVALUATION_FILE,
#     SCORING_FILE,
#     LIGHTDOCK_PDB_FILE,
#     CLUSTER_REPRESENTATIVES_FILE,
# )
# from lightdock.util.logger import LoggingManager
# from lightdock.util.analysis import (
#     read_rmsd_and_contacts_data,
#     read_lightdock_output,
#     write_ranking_to_file,
#     read_cluster_representatives_file,
# )
import logging as log
import numpy as np
from math import sqrt, acos, sin, cos, pi
log.basicConfig(
                    format=
                    '[out]-%(levelname)s:%(message)s'
                    )
CLUSTER_DEFAULT_NAME = "cluster"
DEFAULT_REPRESENTATIVES_EXTENSION = ".repr"
DEFAULT_SWARM_FOLDER = "swarm_"
GSO_OUTPUT_FILE = "gso_%s.out"
EVALUATION_FILE = "evaluation.list"
SCORING_FILE = "scoring.list"
LIGHTDOCK_PDB_FILE = "lightdock_%s.pdb"
CLUSTER_REPRESENTATIVES_FILE = CLUSTER_DEFAULT_NAME + DEFAULT_REPRESENTATIVES_EXTENSION
DEFAULT_ROTATION_STEP = 0.5
RANKING_FILE = "solutions.list"
RANKING_BY_LUCIFERIN_FILE = "rank_by_luciferin.list"
RANKING_BY_RMSD_FILE = "rank_by_rmsd.list"
RANKING_BY_SCORING_FILE = "rank_by_scoring.list"
# log = LoggingManager.get_logger("lgd_rank")

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
    
def read_cluster_representatives_file(cluster_file_name):
    """Reads a LightDock cluster representatives file"""
    with open(cluster_file_name) as fin:
        raw_lines = fin.readlines()
        glowworm_ids = []
        for line in raw_lines:
            line = line.rstrip(os.linesep)
            fields = line.split(":")
            glowworm_id = int(fields[3])
            glowworm_ids.append(glowworm_id)
        return glowworm_ids
    
def parse_coordinates(line):
    """Parses glowworm's coordinates found in line"""
    first = line.index("(")
    last = line.index(")")
    raw = line[first + 1 : last]
    coord = [float(c) for c in raw.split(",")]
    return coord, first, last
    
def write_ranking_to_file(solutions, clashes_cutoff=None, order_by=None):
    """Writes the calculated ranking to a file"""
    if order_by == "luciferin":
        output_file = RANKING_BY_LUCIFERIN_FILE
        solutions.sort(key=operator.attrgetter("luciferin"), reverse=True)
    elif order_by == "scoring":
        output_file = RANKING_BY_SCORING_FILE
        solutions.sort(key=operator.attrgetter("scoring"), reverse=True)
    elif order_by == "rmsd":
        output_file = RANKING_BY_RMSD_FILE
        solutions.sort(key=operator.attrgetter("rmsd"), reverse=False)
    else:
        output_file = RANKING_FILE

    output = open(output_file, "w")
    output.write(
        "Swarm  Glowworm   Coordinates                                             "
        "RecID  LigID  Luciferin  Neigh   VR     RMSD    PDB             Clashes  Scoring\n"
    )
    for solution in solutions:
        if clashes_cutoff:
            if solution.contacts <= clashes_cutoff:
                output.write("%s\n" % (str(solution)))
        else:
            output.write("%s\n" % (str(solution)))
    output.close()

def read_rmsd_and_contacts_data(file_name):
    """Reads a contacts file with columns identified by swarms, glowworm, num_contacts and rmsd"""
    contacts = {}
    rmsds = {}
    if os.path.isfile(file_name):
        with open(file_name) as fin:
            lines = [line.rstrip() for line in fin]
            # Ignore header
            for id_line, line in enumerate(lines[1:]):
                try:
                    fields = line.split()
                    swarm_id = int(fields[0])
                    structure_id = int(fields[1])
                    num_contacts = int(fields[2])
                    rmsd = float(fields[3])
                    if swarm_id in contacts:
                        contacts[swarm_id][structure_id] = num_contacts
                    else:
                        contacts[swarm_id] = {structure_id: num_contacts}
                    if swarm_id in rmsds:
                        rmsds[swarm_id][structure_id] = rmsd
                    else:
                        rmsds[swarm_id] = {structure_id: rmsd}
                except:
                    log.warning("Ignoring line %d in file %s" % (id_line, file_name))
    return contacts, rmsds

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
    parser = argparse.ArgumentParser(prog="lgd_rank")
    parser.add_argument(
        "num_swarms",
        help="number of swarms to consider",
        type=int,
        metavar="num_swarms",
    )
    parser.add_argument("steps", help="steps to consider", type=int, metavar="steps")
    parser.add_argument(
        "-c",
        "--clashes_cutoff",
        help="clashes cutoff",
        dest="clashes_cutoff",
        type=float,
    )
    parser.add_argument(
        "-f",
        "--file_name",
        help="lightdock output file to consider",
        dest="result_file",
    )
    parser.add_argument(
        "--ignore_clusters",
        help="Ignore cluster information",
        dest="ignore_clusters",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        # Parse command line
        args = parse_command_line()

        solutions = []
        contacts = []
        rmsds = []
        if os.path.isfile(EVALUATION_FILE):
            contacts, rmsds = read_rmsd_and_contacts_data(EVALUATION_FILE)

        num_swarms_found = 0
        for swarm_id in range(args.num_swarms):
            if args.result_file:
                result_file_name = os.path.join(
                    DEFAULT_SWARM_FOLDER + str(swarm_id), args.result_file
                )
            else:
                result_file_name = os.path.join(
                    DEFAULT_SWARM_FOLDER + str(swarm_id), (GSO_OUTPUT_FILE % args.steps)
                )

            cluster_representatives_file = os.path.join(
                DEFAULT_SWARM_FOLDER + str(swarm_id), CLUSTER_REPRESENTATIVES_FILE
            )
            clusters = []
            if (
                os.path.isfile(cluster_representatives_file)
                and not args.ignore_clusters
            ):
                clusters = read_cluster_representatives_file(
                    cluster_representatives_file
                )

            scoring_file_name = os.path.join(
                DEFAULT_SWARM_FOLDER + str(swarm_id), SCORING_FILE
            )
            try:
                results = read_lightdock_output(result_file_name)
                num_swarms_found += 1
                for result in results:
                    result.id_swarm = swarm_id
                    result.pdb_file = LIGHTDOCK_PDB_FILE % result.id_glowworm
                    try:
                        result.rmsd = rmsds[result.id_swarm][result.id_glowworm]
                        result.contacts = contacts[result.id_swarm][result.id_glowworm]
                    except Exception:
                        pass
                    if len(clusters):
                        # Clusters read
                        if result.id_glowworm in clusters:
                            solutions.append(result)
                    else:
                        # Default without clustering
                        solutions.append(result)
            except IOError:
                log.warning("Results %s not found, ignoring." % result_file_name)
        # print(solutions)
        write_ranking_to_file(solutions, args.clashes_cutoff)
        write_ranking_to_file(solutions, args.clashes_cutoff, order_by="luciferin")
        write_ranking_to_file(solutions, args.clashes_cutoff, order_by="rmsd")
        write_ranking_to_file(solutions, args.clashes_cutoff, order_by="scoring")

        log.info("Number of swarms: %d" % args.num_swarms)
        log.info("Number of steps: %d" % args.steps)
        if args.clashes_cutoff:
            log.info("Clashes cutoff: %5.3f" % args.clashes_cutoff)
        if args.result_file:
            log.info("Output files: %s" % args.result_file)
        else:
            log.info("Output files: %s" % (GSO_OUTPUT_FILE % args.steps))
        log.info("Number of swarms processed: %d" % num_swarms_found)
        log.info("Done.")

    except KeyboardInterrupt:
        log.info("Caught interrupt...")
        log.info("bye.")
