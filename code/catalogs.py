import numpy as np
import h5py
import fitsio
from astropy import table
import joblib
import pymaster as nmt


class DESY3ShearCat:

    cat_shears = ["1p", "1m", "2p", "2m"]

    @classmethod
    def load_from_pkl(cls, filename):
        return joblib.load(filename)

    def __init__(self, index_file, zbin=None, sample="all", group_name="catalog/metacal",
                 load_cols=["coadd_object_id", "ra", "dec", "e_1", "e_2", "psf_e1", "psf_e2", "weight"],
                 dg=0.01):
        self.index_file = index_file
        self.data_dir = '/'.join(index_file.split('/')[:-1])
        self.dg = dg
        self.zbin = zbin
        self.sample = sample
        if sample == "all":
            self.name = "DES Y3 source catalog" + ('' if zbin is None else f" (z-bin {zbin})")
        else:
            self.name = "DES Y3 source catalog" + ('' if zbin is None else f" (z-bin {zbin}, {sample} sample)")

        self.table_name = f"{group_name}/unsheared"
        self.tables_sheared = {s: f"{group_name}/sheared_{s}" for s in self.cat_shears}

        self.sel_inds = self.get_selection(zbin=zbin, sample=sample)

        # create data table
        with h5py.File(index_file) as index:
            cols = {col: index[self.table_name][col][:][self.sel_inds] for col in load_cols}
        self.data = table.Table(data=cols)

        # calibrate shear
        self.mean_e, self.R = self.calibrate()

    def __repr__(self):
        return f"{self.name}, Ngal = {len(self.data)}"

    def rotate_shear(self):
        """Rotate all shears by random angles."""
        phi = 2*np.pi * np.random.rand(len(self.data))
        c = np.cos(2*phi)
        s = np.sin(2*phi)
        return [c * self.data["g_1"] + s * self.data["g_2"],
                -s * self.data["g_1"] + c * self.data["g_2"]]

    def nmt_catalog(self, lmax, lmax_mask=None, rot=False, psf=False):
        """Setup a NmtFieldCatalog object."""
        if not psf:
            if rot:
                field = self.rotate_shear()
            else:
                field = [self.data["g_1"], self.data["g_2"]]
        else:
            field = [self.data["psf_e1"], self.data["psf_e2"]]
        return nmt.NmtFieldCatalog([self.data["ra"], self.data["dec"]], self.data["weight"],
                                   [-field[0], field[1]], lmax, lmax_mask=lmax_mask, spin=2, lonlat=True)

    def get_selection(self, zbin=None, shear=None, sample="all"):
        """Return indicies of source galaxies within a given selection."""
        samples = ["all", "blue", "red"]
        if sample not in samples:
            raise ValueError(f"sample must be one of {samples}")
        if sample != "all" and zbin is None:
            raise ValueError("color selection is only available for individal redshift bins")
        # look-up table for labelling of redshift bins
        zbins = {i: f"_bin{i}" for i in range(1, 5)} | {None: ''}
        if zbin not in zbins.keys():
            raise ValueError(f"zbin must be one of {zbins.keys()}")
        if shear is None:
            select_name = f"index/select{zbins[zbin]}"
            cat_name = "catalog/metacal/unsheared"
        else:
            if shear not in self.cat_shears:
                raise ValueError(f"shear must be one of {self.cat_shears + [None]}")
            select_name = f"index/select_{shear}{zbins[zbin]}"
            cat_name = f"catalog/metacal/sheared_{shear}"
        with h5py.File(self.index_file) as index:
            inds = index[select_name][:]
            if sample != "all":
                rz_color = -2.5 * np.log10(index[cat_name]["flux_r"][:][inds] /
                                           index[cat_name]["flux_z"][:][inds])
                rz_cuts = [0.5, 0.75, 0.95, 1.3]  # From Table 1 of McCullough et al 2024
                blue_select = rz_color < rz_cuts[zbin-1]
                if sample == "blue":
                    inds = inds[blue_select]
                else:
                    inds = inds[~blue_select]
        return inds

    def get_col(self, col, shear=None):
        pass

    def selection_response(self, diag_only=True):
        """Calculate weighted average of selection response matrix."""
        Rs = []
        # more efficient to first calculate transpose of Rs
        for j in [1, 2]:
            Rs_j = []
            selection_p = self.get_selection(zbin=self.zbin, sample=self.sample, shear=f"{j}p")
            selection_m = self.get_selection(zbin=self.zbin, sample=self.sample, shear=f"{j}m")
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

        mean_e = np.array([np.average(self.data[f"e_{i}"], weights=self.data["weight"]) for i in [1, 2]])
        if verbose:
            print("\tmean e:", mean_e)

        Rg = self.shear_response()
        if verbose:
            print("\tshear response:", np.diag(Rg))
        Rs = self.selection_response()
        if verbose:
            print("\tselection response:", np.diag(Rs))
        R = np.mean(np.diag(Rg + Rs))

        for i in [1, 2]:
            gi = (self.data[f"e_{i}"] - mean_e[i-1]) / R
            self.data[f"g_{i}"] = gi

        return mean_e, R

    def get_dndz(self, product_dir="../datavectors"):
        datavecs = fitsio.FITS(f"{self.data_dir}/{product_dir}/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits")
        z = datavecs["nz_source"]["Z_MID"][:][:-1]
        dndzs = np.array([datavecs["nz_source"][f"BIN{i}"][:] for i in range(1, 5)])
        if self.zbin is not None:
            return z, dndzs[self.zbin-1]

        ngal = np.array([datavecs["nz_source"].read_header()[f"NGAL_{i}"] for i in range(1,5)])
        dndz_total = np.sum([ngal[i] * dndzs[i] for i in range(4)], axis=0) / np.sum(ngal)
        return z, dndz_total
