#####################################################################################################
#
# Interface for communicating with Snp, Bin, and Consideration hubs.
#
#####################################################################################################

import numpy as np
from typing import List, Tuple, Set
from typeguard import typechecked
import numpy.typing as npt
from typing import List

# SortedList is a sorted list implementation in Python for fast insertion and deletion
# https://grantjenks.com/docs/sortedcontainers/sortedlist.html
from sortedcontainers import SortedList

### Static variables
mutation_tries = 20

### General Types

# chromosome number
gen_chrom_num_t = np.uint8
# chromosome snp position
gen_chrom_pos_t = np.uint32
# header snps type
gen_header_snps_t = npt.NDArray[np.str_]
# individual snp type
snp_t = np.str_
# numpy random number generator type
rng_t = np.random.Generator
# r2 score type
r2_t = np.float32

### SNP Hub Types

# bin id position type
snp_hub_bin_t = np.uint16
# header position
snp_hub_pos_t = np.uint32

# generation found/pruned type
snp_hub_gen_t = np.int32

# best individual r2 value type
snp_hub_res_t = np.float32
# best encoder type (in str)
snp_hub_enc_t = np.str_
# corresponding index of the snp position in the bin
snp_hub_idx_t = np.uint16

### Bin Hub Types

# type of object inside bins
bin_hub_arr_t = np.uint32
# type for bin size
bin_hub_size_t = np.uint16
# window distance type
window_distance_t = np.uint32

@typechecked
class SnpHub:
    """
    Interface for communicating with Snp hubs and Bin hubs.
    We do not allow for the creation of new hubs, only the updating of existing ones.
    Treat as a private class.
    """

    class Consideration_Hub:
        """
        Container to maintain snps that have not been pruned by an LD node and have postive marginal r^2.
        """

        def __init__(self, snps: gen_header_snps_t) -> None:
            """
            Create a dictionary with all snps broken down by chromosome and position.
            Then we save them in a dictionary with the chromosome as the key and the snps in a sorted list.
            The individual lists will be updated by removing snps that have been pruned.
            """

            # create a dictionary to hold all snps
            self.consideration_hub = {}

            for snp in snps:
                # split snp up by chromosome and position
                chrom, pos = self.snp_chrm_pos(snp)

                # check if key exists
                if chrom not in self.consideration_hub:
                    self.consideration_hub[chrom] = [pos]
                else:
                    self.consideration_hub[chrom].append(pos)

            # sort all lists within dictionary
            for chrom, pos_l in self.consideration_hub.items():
                # sort the list and store in SortedList and update the dictionary
                pos_l.sort()
                self.consideration_hub[chrom] = SortedList(pos_l)
            return

        # collect all snps for a given chromosome that are not within a certain distance and given anchor position
        def get_snps_in_chrom(self, chrom: gen_chrom_num_t, distance: window_distance_t, anchor: gen_chrom_pos_t) -> List[gen_chrom_pos_t]:
            """
            Get all snps in a given chromosome that are not within a certain distance.
            This is used to get all snps that are not pruned by an LD node.
            """

            # make sure the chromosome exists
            assert chrom in self.consideration_hub

            # get all positions in the chromosome
            pos_l = self.consideration_hub[chrom]

            # return all positions that are not within the distance
            return [pos for pos in pos_l if abs(np.int32(pos) - np.int32(anchor)) > np.int32(distance)]

        # remove snp from the consideration_hub dictionary
        def remove_snp(self, snp: snp_t) -> None:
            # get chromosome and position
            chrom, pos = self.snp_chrm_pos(snp)

            # make sure the chromosome exists
            assert chrom in self.consideration_hub

            # remove snp from the list
            self.consideration_hub[chrom].remove(pos)

            return

        # print all snps in the consideration_hub dictionary
        def print_hub(self) -> None:
            for chrom, pos_l in self.consideration_hub.items():
                print(f"Chromosome: {chrom}")
                print(f"Positions: {pos_l}")
            return

        # helper to generate chromosome number and snp position
        def snp_chrm_pos(self, snp: snp_t) -> Tuple[gen_chrom_num_t, gen_chrom_pos_t]:
            chrom, pos = snp.split('.')
            chrom, pos = gen_chrom_num_t(chrom), gen_chrom_pos_t(pos)
            return chrom, pos

        # get a random snp from the hub
        def get_ran_snp(self, rng_: rng_t) -> snp_t:
            # make sure we have at least one snp
            assert self.get_total() > 0

            # initialize rng
            rng = np.random.default_rng(rng_)
            chrom = []

            for chrm in list(self.consideration_hub.keys()):
                if len(self.consideration_hub[chrm]) > 0:
                    chrom.append(chrm)


            # get a random chromosome key
            chrom = rng.choice(chrom)

            # get a random bin index from the chromosome
            position = rng.choice(self.consideration_hub[chrom])

            return snp_t(f"{chrom}.{position}")

        # get total number of items in the consideration_hub dictionary
        def get_total(self) -> np.uint32:
            sum = 0
            for _, pos_l in self.consideration_hub.items():
                sum += len(pos_l)
            return np.uint32(sum)

        # get list of keys with at least one snp
        def get_keys_with_snps(self) -> List[gen_chrom_num_t]:
            keys = []
            for k, v in self.consideration_hub.items():
                if len(v) > 0:
                    keys.append(k)
            return keys

    class Hub:
        """
        Hub to keep track of snp weights.
        """
        def __init__(self):
            """
            self.hub: dictionary to hold all snp and values
            assuming that all snps are already in the hub
            if we get a snp that is not in the hub, we throw an error in debug mode

                   res_pos = 0 # position for r2 recived from evaluation
                   idx_pos = 1 # position for bin number in hub value list
                   pos_pos = 2 # position for header position in hub value list
                   enc_pos = 3 # position for the corresponding encoder types in hub value list
                  seen_pos = 4 # position for the seen flag in hub value list
                pruned_pos = 5 # position for the pruned flag in hub value list
              gen_seen_pos = 6 # position for the general seen flag in hub value list
            gen_pruned_pos = 7 # position for the general pruned flag in hub value list
            """

            # {snp: [res(np.float32),bin(np.uint32),idx(np.uint32),
            # pos(np.uint32),enc(np.str_),seen(bool)], prunned(bool),
            # gen_seen(np.unit32), gen_prunned(np.unit32)}
            self.hub = {}

        # will add snp, sum, bin, pos， idx, res, typ to the hub
        def add_to_hub(self,
                       snp: snp_t,
                       res: snp_hub_res_t,
                       idx: snp_hub_idx_t,
                       pos: snp_hub_pos_t,
                       enc: snp_hub_enc_t,
                       seen=False,
                       prunned=False,
                       gen_seen: snp_hub_gen_t=snp_hub_gen_t(-1),
                       gen_pruned:snp_hub_gen_t=snp_hub_gen_t(-1)) -> None:
            """
            will take in a snp, sum, cnt, bin, and pos and add it to the hub

            Args:
                (k) snp (snp_t): chrm.pos string
                (0) res (snp_hub_res_t): best r2 result placeholder, default = -1
                (1) idx (snp_hub_idx_t): corresponding index of the snp position in the bin
                (2) pos (snp_hub_pos_t): position of the snp in the csv header
                (3) enc (snp_hub_enc_t): best encoder type placeholder, default = ''
                (4) seen (bool): has this snp been seen before, default = False
                (5) prunned (bool): has this snp been prunned, default = False
                (6) gen_seen (snp_hub_gen_t): generation seen, default = -1
                (7) gen_prunned (snp_hub_gen_t): generation prunned, default = -1
            """

            # add to hub
            self.hub[snp] = [res,idx,pos,enc,seen,prunned,gen_seen,gen_pruned]
            return

        # get snp result r^2
        def get_uni_res(self, snp: snp_t) -> snp_hub_res_t:
            # check for snp existence
            assert snp in self.hub
            # return data
            return np.float32(self.hub[snp][0])

        # get snp position idx in the bin
        def get_snp_idx(self, snp: snp_t) -> snp_hub_idx_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return snp_hub_idx_t(self.hub[snp][1])

        # get snp position in csv header
        def get_snp_pos(self, snp: snp_t) -> snp_hub_pos_t:
            # assert that snp is in hub
            assert snp in self.hub
            # return data
            return snp_hub_pos_t(self.hub[snp][2])

        # get snp encoder type
        def get_uni_encoding(self, snp: snp_t) -> snp_hub_enc_t:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][3]

        # has this snp been seen before
        def has_been_seen(self, snp: snp_t) -> bool:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][4]

        # has this snp been pruned
        def has_been_pruned(self, snp: snp_t) -> bool:
            # check snp exists in the hub
            assert snp in self.hub
            # return the type
            return self.hub[snp][5]

        # flip the pruned flag
        def flip_prunned(self, snp: snp_t, gen_pruned: snp_hub_gen_t) -> None:
            # check snp exists in the hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.has_been_pruned(snp) == False

            # flip the flag
            self.hub[snp][5] = True
            # record the generation prunned
            self.hub[snp][7] = gen_pruned

            return

        # flip the seen flag
        def flip_seen(self, snp: snp_t) -> None:
            # check snp exists in the hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.has_been_seen(snp) == False

            # flip the flag
            self.hub[snp][4] = True
            return

        # update snp hub with the r2 and encoding type
        # assuming that this only gets called once per snp
        # NEW: problem_type parameter to have different thresholds for classification/regression
        def update_snp(self, snp: snp_t, value: np.float32, encoding:np.str_, gen_seen: snp_hub_gen_t, problem_type: str) -> None:
            # assert that snp is in hub
            assert snp in self.hub
            # make sure we have not seen this snp before
            assert self.has_been_seen(snp) == False

            # flip the seen flag
            self.flip_seen(snp)
            # update the results
            self.hub[snp][0] = value
            # update the encoder type
            self.hub[snp][3] = encoding
            # update the generation seen
            self.hub[snp][6] = gen_seen

            if problem_type == "regression":
                # if value is negative, set as prunned
                if value < 0.0:
                    self.flip_prunned(snp, gen_seen)
            elif problem_type == "classification":
                if value < 0.0001:
                    self.flip_prunned(snp, gen_seen)

            return

    class Ordered_Hub:
        """
        Hub to all snps partitioned by their chromosome, and each chromosome contains the sorted order of the positions.
        """
        def __init__(self) -> None:
            # {chrom: np.array([pos1, pos2, ...]), dtype=gen_chrom_pos_t)}]}
            # pos1 < pos2 < ... < posN
            self.order = {}
            return

        def get_bin_size(self, chrom: gen_chrom_num_t, bin: bin_hub_size_t) -> bin_hub_size_t:
            # make sure the chromosome exists
            assert chrom in self.order

            # make sure the bin exists
            assert bin < len(self.bins[chrom])
            assert bin >= 0

            # return the size of the bin
            return bin_hub_size_t(len(self.bins[chrom][bin]))

        def get_pos_in_bin(self, chrom: gen_chrom_num_t, bin: bin_hub_size_t, idx: bin_hub_size_t) -> gen_chrom_pos_t:
            # make sure the chromosome exists
            assert chrom in self.bins
            # make sure the bin exists
            assert bin < len(self.bins[chrom])
            assert bin >= 0
            # make sure the index is within the bin
            assert idx < len(self.bins[chrom][bin])
            assert idx >= 0

            # return the position in the bin
            return gen_chrom_pos_t(self.bins[chrom][bin][idx])

        # get window for a given chromosome and position index within a given distance
        def get_window(self, chrom: gen_chrom_num_t, idx: snp_hub_idx_t, distance: np.uint32) -> List[gen_chrom_pos_t]:
            # make sure the chromosome exists
            assert chrom in self.order
            # make sure idx is within the bounds of the order dictionary for a chromosome
            assert idx < len(self.order[chrom])
            assert idx >= 0

            # get all positions that fall within the distance from the given index
            window_list = []
            anchor = self.order[chrom][idx]

            # get all positions to the left of the anchor position
            left_bound = np.int32(idx) - 1

            if left_bound >= 0:
                while abs(anchor - self.order[chrom][left_bound]) <= distance:
                    window_list.append(self.order[chrom][left_bound])
                    left_bound -= 1

                    if left_bound < 0:
                        break

            # get all positions to the right of the anchor position
            right_bound = np.int32(idx) + 1

            if right_bound < len(self.order[chrom]):
                while abs(anchor - self.order[chrom][right_bound]) <= distance:
                    window_list.append(self.order[chrom][right_bound])
                    right_bound += 1

                    if right_bound >= len(self.order[chrom]):
                        break

            return window_list

        def count_order_objs(self) -> np.uint32:
            """
            Count the total number of bin objects in the order dictionary.
            This is used to ensure that all snps are accounted for.
            """
            count = 0
            for _, order in self.order.items():
                count += len(order)
            return np.uint32(count)

        # create ordered lists for each chromosome based on the snps provided
        def generate_order(self, snps: gen_header_snps_t) -> List:
            # quick checks
            assert len(snps) > 0

            # load all snps into the order dictionary with chromosome as key and the values as lists of positions
            for snp in snps:
                chrom, pos = self.snp_chrm_pos(snp)
                if chrom not in self.order:
                    self.order[chrom] = []
                self.order[chrom].append(pos)

            # sort all lists within the order dictionary
            for chrom, order in self.order.items():
                self.order[chrom] = sorted(order)

            # collect each snps bin number
            snp_bins = []
            # go through self.order and collect all snps, bin_num
            for chrom, order in self.order.items():
                for idx, pos in enumerate(order):
                    snp_bins.append((np.str_(f"{chrom}.{pos}"), idx))

            # cast all bins to list for search efficiency
            for chrom, o in self.order.items():
                self.order[chrom] = [gen_chrom_pos_t(p) for p in o]

            # make sure all SNPs are accounted for
            assert len(snps) == self.count_order_objs()
            # make sure snp_bins is the correct size
            assert len(snp_bins) == len(snps)

            return snp_bins # [(snp, bin_num, idx_in_bin), ...]

        # helper to generate chromosome number and snp position
        def snp_chrm_pos(self, snp: snp_t) -> Tuple[gen_chrom_num_t, gen_chrom_pos_t]:
            chrom, pos = snp.split('.')
            chrom, pos = gen_chrom_num_t(chrom), gen_chrom_pos_t(pos)
            return chrom, pos

    # initialize all hubs
    def __init__(self, snps: gen_header_snps_t) -> None:
        print('Initializing Snp Hub')

        # initialize non pruned hub
        print('Initializing Consideration Hub')
        self.consideration_hub = self.Consideration_Hub(snps)
        print('Consideration Hub Initialized\n')

        # bin hub stuff
        print('Initializing Ordered Hub')
        self.order = self.Ordered_Hub()
        # get snps and their bin id
        snp_bin = self.order.generate_order(snps)
        print('Order Hub Initialized')

        # snp hub stuff
        self.hub = self.Hub()
        # update snp_hub with snp_bin and snp header positions
        for s in snp_bin:
            # find where snp is located in csv header (snps)
            h_pos = snp_hub_pos_t(np.where(snps == s[0])[0][0])

            # make sure that the snp is in the correct format
            assert isinstance(s[0], np.str_)
            # make sure header position matches the csv header
            assert s[0] == snps[h_pos]
            # make sure the snp bin poisition is correct
            assert self.order.snp_chrm_pos(s[0])[1] == self.order.order[self.order.snp_chrm_pos(s[0])[0]][s[1]]

            # add snp to hub with all its data
            self.hub.add_to_hub(snp=s[0],
                                res=snp_hub_res_t(-1.0),
                                idx=snp_hub_idx_t(s[1]),
                                pos=h_pos,
                                enc=snp_hub_enc_t(''),
                                seen=False,
                                prunned=False,
                                gen_seen=snp_hub_gen_t(-1),
                                gen_pruned=snp_hub_gen_t(-1))
        print('SNP Hub Initialized')
        return

    # get best type of encoder for a given snp
    def get_uni_encoding(self, snp: snp_t) -> snp_hub_enc_t:
        return self.hub.get_uni_encoding(snp)

    # get r2 for a given snp from snp hub
    def get_uni_res(self, snp: snp_t) -> snp_hub_res_t:
        return self.hub.get_uni_res(snp)

    # save the epi_hub and snp_hub to a file
    def save_hubs(self, save_dir: str) -> None:
        """
        (k) snp (snp_t): chrm.pos string
        (0) res (snp_hub_res_t): best r2 result placeholder, default = -1
        (1) idx (snp_hub_idx_t): corresponding index of the snp position in the bin
        (2) pos (snp_hub_pos_t): position of the snp in the csv header
        (3) enc (snp_hub_enc_t): best encoder type placeholder, default = ''
        (4) seen (bool): has this snp been seen before, default = False
        (5) prunned (bool): has this snp been prunned, default = False
        (6) gen_seen (snp_hub_gen_t): generation seen, default = -1
        (7) gen_prunned (snp_hub_gen_t): generation prunned, default = -1
        """

        # Save snp hub with headers
        # for everything in row 0 append chr
        snp_data = []
        for k, v in self.hub.hub.items():
            # add chr to k here
            # k: snp (row[0])
            # v[0]: r2 (row[1])
            # v[1]: idx (row[2])
            # v[2]: pos (row[3])
            # v[3]: enc (row[4])
            # v[4]: seen (row[5])
            # v[5]: prunned (row[6])
            # v[6]: gen_seen (row[7])
            # v[7]: gen_prunned (row[8])
            snp_data.append([k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]])

        # Sort snp_data by the second column (AVG_R2)
        snp_data.sort(key=lambda x: x[1], reverse=True)  # reverse=True for descending order

        # Write snp hub to file
        with open(save_dir+"snp_hub.csv", 'w') as f:
            # Write the headers for the snp_file
            f.write("snp,chr,bp,r2,position,encoding,seen,pruned,gen_seen,gen_pruned\n")
            for row in snp_data:
                # split snp into chromosome and position
                chrom, pos = row[0].split('.')
                # f.write(f"{row[0]},{chrom},{pos},{row[1]},{row[2]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]}\n")
                # NEW PT2: ADD THESE 2 LINES:
                snp_name = f"chr{row[0]}"
                # f.write(f"{snp_name},{chrom},{pos},{row[1]},{row[2]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]},{row[9]}\n")
                # Removing row[9] for now because didn't have that above:
                f.write(f"{snp_name},{chrom},{pos},{row[1]},{row[2]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]}\n")
                

        # save csv with both seen and not prunned snps
        # Write snp hub to file
        with open(save_dir+"consideration_hub.csv", 'w') as f:
            # Write the headers for the snp_file
            f.write("snp,r2,encoding\n")
            for row in snp_data:
                if row[6] == True and row[7] == False:
                    # NEW PT2: ADD THIS:
                    snp_name = f"chr{row[0]}"
                    # and then insert snp_name for row[0]
                    # f.write(f"{row[0]},{row[1]},{row[4]}\n")
                    f.write(f"{snp_name},{row[1]},{row[4]}\n")
        return

    # update snp hub with best univariate r2 result and corresponding encoder type
    def update_snp_hub(self, snp:snp_t, result:snp_hub_res_t, type: snp_hub_enc_t, gen_seen: snp_hub_gen_t, problem_type: str) -> None:
        # update Hub object: if r2 is negative, flip prunned flag
        self.hub.update_snp(snp, result, type, gen_seen, problem_type)
        if problem_type == "regression":
            # update Consideration_Hub object: if r2 is negative, remove snp from non prunned
            if result < r2_t(0.0):
                self.consideration_hub.remove_snp(snp)
        elif problem_type == "classification":
            if result < r2_t(0.0001):
                self.consideration_hub.remove_snp(snp)
        return

    # check if snp has encoder type recorded in the snp hub
    def is_encoder_in_hub(self, snp:snp_t) -> bool:
        return self.hub.has_been_seen(snp)

    # get snp position from the snp hub
    def get_snp_pos(self, snp: snp_t) -> snp_hub_pos_t:
        return self.hub.get_snp_pos(snp)

    # get a snp from the same chromosome and bin with r2 > 0.0 based on r2 weight
    def get_smt_snp_in_bin(self, snp: snp_t, rng_: rng_t, window_distance: window_distance_t, problem_type: str) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # break snp into chromosome and position
        chrom, _ = self.order.snp_chrm_pos(snp)

        # get window for the snp
        window = self.order.get_window(chrom, self.hub.get_snp_idx(snp), window_distance)

        # reduce the window to only snps that have r2 > 0.0 and not prunned
        valid_snps = []
        r2_list = []
        for pos in window:
            s = snp_t(f"{chrom}.{pos}")

            # r2 > 0.0 and seen and not pruned
            if problem_type == "regression":
                if self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s) and self.hub.has_been_pruned(s) == False:
                    valid_snps.append(s)
                    r2_list.append(self.hub.get_uni_res(s))
            elif problem_type == "classification":
                if self.hub.get_uni_res(s) > r2_t(0.0001) and self.hub.has_been_seen(s) and self.hub.has_been_pruned(s) == False:
                    valid_snps.append(s)
                    r2_list.append(self.hub.get_uni_res(s))

        # if no snps were returned, return a random one
        if len(valid_snps) == 0:
            return self.consideration_hub.get_ran_snp(rng)

        # normalize r2_list to sum to 1
        r2_list = np.array(r2_list, dtype=np.float32)
        r2_list /= np.sum(r2_list)

        # get a random snp based on r2 scores as weights
        choice = rng.choice(valid_snps, p=r2_list)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(valid_snps, p=r2_list)

        # get a random snp based on r2 scores as weights
        return snp

    # get a random snp from the same chromosome and bin
    def get_ran_snp_in_bin(self, snp: snp_t, rng_: rng_t, window_distance: window_distance_t, problem_type: str) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # break snp into chromosome and position
        chrom, _ = self.order.snp_chrm_pos(snp)

        # get window for the snp
        window = self.order.get_window(chrom, self.hub.get_snp_idx(snp), window_distance)

        # collect all snps that have (not pruned and seen) or (r2 > 0.0 and seen)
        snps = []
        for p in window:
            s = snp_t(f"{chrom}.{p}")

            # not seen
            not_seen = self.hub.has_been_seen(s) == False
            if problem_type == "regression":
                # r2 > 0.0 and seen and not pruned
                seen_r2_np = self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s) and self.hub.has_been_pruned(s) == False
            elif problem_type == "classification":
                seen_r2_np = self.hub.get_uni_res(s) > r2_t(0.0001) and self.hub.has_been_seen(s) and self.hub.has_been_pruned(s) == False

            if (not_seen or seen_r2_np) and s != snp:
                snps.append(s)

        # if no snps were collected, return a random snp from non pruned
        if len(snps) == 0:
            return self.get_random_non_pruned_snp(rng)

        # try to get a random snp position that is not the same as the input snp
        choice = rng.choice(snps)

        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps)

        # get a random snp based
        return snp

    # geta snp from the same chromosome but different bin
    def get_smt_snp_in_chrm(self, snp: snp_t, rng_: rng_t, window_distance: window_distance_t, problem_type: str) -> snp_t:
        # initialize rng
        rng = np.random.default_rng(rng_)

        # break snp into chromosome and position
        chrom, position = self.order.snp_chrm_pos(snp)

        # get bin id for the snp
        snps_in_chrom = self.consideration_hub.get_snps_in_chrom(chrom, window_distance, position)

        # collect all snps that have not been pruned and have r2 > 0.0
        snps = []
        r2 = []

        # loop through all non prunned snps and collect the ones with r2 > 0.0 and not pruned
        for pos in snps_in_chrom:
            s = snp_t(f"{chrom}.{pos}")
            assert s != snp, "SNP should not be the same as the input SNP"

            if problem_type == "regression":
                if self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s):
                    snps.append(s)
                    r2.append(self.hub.get_uni_res(s))
            elif problem_type == "classification":
                if self.hub.get_uni_res(s) > r2_t(0.0001) and self.hub.has_been_seen(s):
                    snps.append(s)
                    r2.append(self.hub.get_uni_res(s))

        r2 /= np.sum(r2, dtype=np.float32)

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.get_ran_snp_in_chrm(snp, rng, window_distance, problem_type)

        # get a random snp based on r2 scores as weights
        choice = rng.choice(snps, p=r2)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps, p=r2)

        # get a random snp based on r2 scores as weights
        return snp

    # get a random snp from the same chromosome but different bin
    def get_ran_snp_in_chrm(self, snp: snp_t, rng_: rng_t, window_distance: window_distance_t, problem_type: str) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # break snp into chromosome and position
        chrom, position = self.order.snp_chrm_pos(snp)

        # get bin id for the snp
        snps_in_chrom = self.consideration_hub.get_snps_in_chrom(chrom, window_distance, position)

        # collect all snps that have not been pruned and have r2 > 0.0
        snps = []

        # loop through all non prunned snps and collect the ones with r2 > 0.0 and not pruned
        for pos in snps_in_chrom:
            # make snps
            s = snp_t(f"{chrom}.{pos}")
            # not seen
            not_seen = self.hub.has_been_seen(s) == False
            if problem_type == "regression":
                # r2 > 0.0 and seen and not pruned
                seen_r2_np = self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s) and self.hub.has_been_pruned(s) == False
            elif problem_type == "classification":
                seen_r2_np = self.hub.get_uni_res(s) > r2_t(0.0001) and self.hub.has_been_seen(s) and self.hub.has_been_pruned(s) == False

            if not_seen or seen_r2_np:
                snps.append(s)

        # if no snps were collected, return a random snp from non pruned
        if len(snps) == 0:
            return self.get_random_non_pruned_snp(rng)

        # try to get a random snp that is not the same as the input snp
        choice = rng.choice(snps)

        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps)

        # return same snp
        return snp

    # get a snp from outside the chromosome with r2 > 0.0 based on r2 weight
    def get_smt_snp_out_chrm(self, snp: snp_t, rng_: rng_t, problem_type: str) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # set the random number generator
        rng = np.random.default_rng(rng_)

        # split up the snp into chromosome and position
        chrom, pos = self.consideration_hub.snp_chrm_pos(snp)

        # get keys for non pruned snps
        chrom_keys = self.consideration_hub.get_keys_with_snps()

        # remove the current chromosome from the list
        if chrom in chrom_keys:
            chrom_keys.remove(chrom)

        # randomly select a chromosome
        c_pic = rng.choice(chrom_keys)

        # collect all snps that have not been pruned and have r2 > 0.0
        snps = []
        r2 = []

        # loop through all non pruned snps and collect the ones with r2 > 0.0 and not pruned
        for pos in self.consideration_hub.consideration_hub[c_pic]:
            s = snp_t(f"{c_pic}.{pos}")

            if problem_type == "regression":
                if self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s):
                    snps.append(s)
                    r2.append(self.hub.get_uni_res(s))
            elif problem_type == "classification":
                if self.hub.get_uni_res(s) > r2_t(0.0001) and self.hub.has_been_seen(s):
                    snps.append(s)
                    r2.append(self.hub.get_uni_res(s))

        r2 = r2 / np.sum(r2, dtype=np.float32)

        # if no snps were returned, return a random one
        if len(snps) == 0:
            return self.get_ran_snp_out_chrm(snp, rng, problem_type)

        # get a random snp based on r2 scores as weights
        choice = rng.choice(snps, p=r2)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps, p=r2)

        # get a random snp based on r2 scores as weights
        return snp

    # get random snp from outside the chromosome
    def get_ran_snp_out_chrm(self, snp: snp_t, rng_: rng_t, problem_type: str) -> snp_t:
        # make sure there is a '.' inside the snp string
        assert '.' in snp

        # initialize rng
        rng = np.random.default_rng(rng_)

        # split up the snp into chromosome and position
        chrom, pos = self.consideration_hub.snp_chrm_pos(snp)

        # get keys for non pruned snps
        chrom_keys = self.consideration_hub.get_keys_with_snps()

        # remove the current chromosome from the list
        if chrom in chrom_keys:
            chrom_keys.remove(chrom)

        # randomly select a chromosome
        c_pic = rng.choice(chrom_keys)

        # collect all snps that have not been pruned and have r2 > 0.0
        snps = []

        # loop through all non pruned snps and collect them
        for pos in self.consideration_hub.consideration_hub[c_pic]:
            # make snps
            s = snp_t(f"{c_pic}.{pos}")
            # not seen
            not_seen = self.hub.has_been_seen(s) == False
            if problem_type == "regression":
                # r2 > 0.0 and seen and not pruned
                seen_r2_np = self.hub.get_uni_res(s) > r2_t(0.0) and self.hub.has_been_seen(s) and self.hub.has_been_pruned(s) == False
            elif problem_type == "classification":
                seen_r2_np = self.hub.get_uni_res(s) > r2_t(0.0001) and self.hub.has_been_seen(s) and self.hub.has_been_pruned(s) == False

            if (not_seen or seen_r2_np):
                snps.append(s)

        # if no snps were collected, return a random snp from non pruned
        if len(snps) == 0:
            return self.get_random_non_pruned_snp(rng)

        # try to get a random snp that is not the same as the input snp
        choice = rng.choice(snps)

        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = rng.choice(snps)

        return snp

    # get a random snp from all possible snps
    def get_ran_snp(self, rng_: rng_t, snp = None) -> snp_t:
        # initialize rng
        rng = np.random.default_rng(rng_)

        # if snp is None, return a random snp
        # None means we don't need to check if the snp is the same as the input snp
        if snp is None:
            return self.consideration_hub.get_ran_snp(rng)

        # if snp is provided, return a random one that is not the same as the input snp
        choice = self.consideration_hub.get_ran_snp(rng)

        # try to get a random snp that is not the same as the input snp
        for _ in range(mutation_tries):
            if choice != snp:
                return choice
            choice = self.consideration_hub.get_ran_snp(rng)

        return choice

    def get_k_snps_from_chrom(self, rng_:rng_t, chrom:gen_chrom_num_t, k:int) -> Set[snp_t]:
        # initialize rng and set
        rng = np.random.default_rng(rng_)
        k_snps = set()

        # make sure the chrom is not out of bound
        assert chrom in self.order.order

        # get random positions from chromosome's position lists
        while len(k_snps) < k: # to make sure there are no replicates
            # sample a random position from the chromosome's position list
            k_snps.add(snp_t(f"{chrom}.{rng.choice(self.order.order[chrom])}"))
        # return the snp set
        return k_snps

    def does_snp_exist(self, snp: snp_t) -> bool:
        return snp in self.hub.hub

    # has snp been pruned?
    def has_been_pruned(self, snp: snp_t) -> bool:
        return self.hub.has_been_pruned(snp)

    # process the prunned snps
    def process_pruned_snps(self, snps: Set[snp_t], gen_pruned: snp_hub_gen_t) -> None:
        # go through each snp and update the hub
        for snp in snps:
            # check to make sure we have not prunned this snp before
            assert self.hub.has_been_pruned(snp) == False

            # flip snp to pruned
            self.hub.flip_prunned(snp, gen_pruned)

            # delete snp from non pruned
            self.consideration_hub.remove_snp(snp)

    # function to take in a list of snps and generate a dictionary of snps and their corresponding r2 values
    def generate_r2_dict(self, snps: Set[snp_t]) -> List:
        # make sure
        assert len(snps) > 0

        return [(snp, self.get_uni_res(snp)) for snp in snps]

    # check if all snps have been pruned
    def all_snps_pruned(self, snps: Set[snp_t]) -> bool:

        # if any snps has not been prunned return False
        for snp in snps:
            if self.hub.has_been_pruned(snp) == False:
                return False

        # return true if all snps have been pruned
        return True

    # print size of non pruned hub
    def consideration_hub_size(self) -> np.uint32:
        return self.consideration_hub.get_total()

    # return a random snp from the non pruned hub
    def get_random_non_pruned_snp(self, rng_: rng_t) -> snp_t:
        rng = np.random.default_rng(rng_)
        return self.consideration_hub.get_ran_snp(rng)

    def get_keys_with_snps(self) -> List[gen_chrom_num_t]:
        return self.consideration_hub.get_keys_with_snps()

    # count number of unseen snps in the hub from non_pruned object
    def seen_snps_proportion(self) -> None:
        count = 0
        for chrm in self.consideration_hub.consideration_hub:
            for pos in self.consideration_hub.consideration_hub[chrm]:
                # count if the snp has been seen
                if self.hub.has_been_seen(snp_t(f"{chrm}.{pos}")) == False:
                    count += 1

        # print proportion of unseen snps
        print(f"Proportion of seen SNPs: {1.0 - (count/len(self.hub.hub)):.2%}", flush=True)
        return