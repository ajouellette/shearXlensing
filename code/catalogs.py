import numpy as np
import h5py
from astropy import table


class DESY3ShearCat:

    cat_shears = ["1p", "1m", "2p", "2m"]

    def __init__(self, index_file, zbin=None, group_name="catalog/metacal",
                 load_cols=["coadd_object_id", "ra", "dec", "e_1", "e_2", "weight"], dg=0.01):
        self.index_file = index_file
        self.dg = dg
        self.zbin = zbin
        self.name = "DES Y3 source catalog" + '' if zbin is None else f" (z-bin {zbin})"

        self.table_name = f"{group_name}/unsheared"
        self.tables_sheared = {s: f"{group_name}/sheared_{s}" for s in self.cat_shears}

        self.sel_inds = self.get_selection(zbin=zbin)

        with h5py.File(index_file) as index:
            cols = {col: index[self.table_name][col][:][self.sel_inds] for col in load_cols}
        self.data = table.Table(data=cols)

    def get_selection(self, zbin=None, shear=None):
        """Return indicies of source galaxies within a given selection."""
        # look-up table for labelling of redshift bins
        zbins = {i: f"_bin{i}" for i in range(1, 5)} | {None: ''}
        if zbin not in zbins.keys():
            raise ValueError(f"zbin must be one of {zbins.keys()}")
        if shear is None:
            shear = ''
        else:
            if shear not in self.cat_shears:
                raise ValueError(f"shear must be one of {self.cat_shears + [None]}")
            shear = '_' + shear
        select_name = f"index/select{shear}{zbins[zbin]}"
        with h5py.File(self.index_file) as index:
            inds = index[select_name][:]
        return inds

    def get_col(self, col, shear=None):
        pass

    def selection_response(self, diag_only=True):
        """Calculate weighted average of selection response matrix."""
        Rs = []
        # more efficient to first calculate transpose of Rs
        for j in [1, 2]:
            Rs_j = []
            selection_p = self.get_selection(zbin=self.zbin, shear=f"{j}p")
            selection_m = self.get_selection(zbin=self.zbin, shear=f"{j}m")
            with h5py.File(self.index_file) as index:
                weights_p = index[self.tables_sheared[f"{j}p"]]["weight"][:][selection_p]
                weights_m = index[self.tables_sheared[f"{j}m"]]["weight"][:][selection_m]
            for i in [1, 2]:
                if diag_only and i != j:
                    Rs_j.append(0)
                else:
                    with h5py.File(self.index_file) as index:
                        ei = index[self.table_name][f"e_{i}"][:]
                    mean_ei_sp = np.average(ei[selection_p], weights=weights_p)
                    mean_ei_sm = np.average(ei[selection_m], weights=weights_m)
                    Rs_j.append((mean_ei_sp - mean_ei_sm) / (2*self.dg))
            Rs.append(Rs_j)
        Rs = np.array(Rs).T
        return Rs

    def shear_response(self, diag_only=True):
        """Calculate weighted average of shear response matrix."""
        Rg = []
        for i in [1, 2]:
            Rg_i = []
            for j in [1, 2]:
                if diag_only and i != j:
                    R_ij = 0
                else:
                    with h5py.File(self.index_file) as index:
                        R_ij = index[self.table_name][f"R{i}{j}"][:][self.sel_inds]
                    R_ij = np.average(R_ij, weights=self.data["weight"])
                Rg_i.append(R_ij)
            Rg.append(Rg_i)
        Rg = np.array(Rg)
        return Rg

    def calibrate(self, subtract_mean=True, verbose=True):
        """Calibrate elipticity response and calculate reduced shear."""

        if "g_1" in self.data.colnames and "g_2" in self.data.colnames:
            print("Elipticity response already calibrated")
            return [0,0], 1

        mean_e = [np.average(self.data[f"e_{i}"], weights=self.data["weight"]) for i in [1, 2]]
        if verbose:
            print("mean e:", mean_e)

        Rg = self.shear_response()
        if verbose:
            print("shear response:", np.diag(Rg))
        Rs = self.selection_response()
        if verbose:
            print("selection response:", np.diag(Rs))
        R = np.mean(np.diag(Rg + Rs))

        for i in [1, 2]:
            gi = (self.data[f"e_{i}"] - mean_e[i-1]) / R
            self.data[f"g_{i}"] = gi

        return mean_e, R

    def get_dndz(self):
        pass
