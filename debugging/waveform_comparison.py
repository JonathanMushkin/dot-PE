import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from cogwheel import skyloc_angles, waveform
from cogwheel.gw_utils import DETECTORS, get_geocenter_delays, get_fplus_fcross_0
from cogwheel.utils import read_json
from lal import GreenwichMeanSiderealTime
from dot_pe import config
from dot_pe.likelihood_calculating import LinearFree
from dot_pe.single_detector import SingleDetectorProcessor

print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

cache_dir = Path(__file__).parent / "posterior_cache"
coherent_posterior = read_json(cache_dir / "Posterior.json")
event_data = coherent_posterior.likelihood.event_data

bank_folder = Path("test_bank")
bank_config = read_json(bank_folder / "bank_config.json")
bank_config["fbin"] = np.array(bank_config["fbin"])
bank_config["m_arr"] = np.array(bank_config["m_arr"])
intrinsic_bank_df = pd.read_feather(bank_folder / "intrinsic_sample_bank.feather")

n_phi = 32
bank_file_path = bank_folder / "intrinsic_sample_bank.feather"
waveform_dir = bank_folder / "waveforms"

event_data_1d = copy.deepcopy(event_data)
det_name = event_data.detector_names[0]
det_indices = [event_data_1d.detector_names.index(det) for det in list(det_name)]

array_attributes = ["strain", "blued_strain", "wht_filter"]
for attr in array_attributes:
    setattr(event_data_1d, attr, getattr(event_data_1d, attr)[det_indices])

tuple_attributes = ["detector_names"]
for attr in tuple_attributes:
    temp = tuple(np.take(getattr(event_data_1d, attr), det_indices))
    setattr(event_data_1d, attr, temp)

par_dic_0 = coherent_posterior.likelihood.par_dic_0.copy()
wfg_1d = waveform.WaveformGenerator.from_event_data(
    event_data_1d, bank_config["approximant"]
)
likelihood_linfree = LinearFree(event_data_1d, wfg_1d, par_dic_0, bank_config["fbin"])

sdp = SingleDetectorProcessor(
    bank_file_path,
    waveform_dir,
    n_phi,
    bank_config["m_arr"],
    likelihood_linfree,
)

wfg = waveform.WaveformGenerator.from_event_data(event_data, bank_config["approximant"])

i = 0
intrinsic_params = intrinsic_bank_df.iloc[i].to_dict()
intrinsic_params["f_ref"] = bank_config["f_ref"]

waveform_par_dic_0 = intrinsic_params | config.DEFAULT_PARAMS_DICT
h_mpb_from_wfg = wfg.get_hplus_hcross(
    bank_config["fbin"], waveform_par_dic_0, by_m=True
)

amp, phase = sdp.intrinsic_sample_processor.load_amp_and_phase(
    bank_folder / "waveforms", np.array([i])
)
h_mpb_from_bank = (amp * np.exp(1j * phase))[0]

dt = sdp.intrinsic_sample_processor.cached_dt_linfree_relative[i]
print(f"dt: {dt}")

timeshift = np.exp(2j * np.pi * dt * bank_config["fbin"])
h_mpb_from_wfg *= timeshift[None, None, :]

print(f"h_mpb_from_wfg shape: {h_mpb_from_wfg.shape}")
print(f"h_mpb_from_bank shape: {h_mpb_from_bank.shape}")
print(f"h_mpb_from_wfg magnitude: {np.abs(h_mpb_from_wfg).max():.6e}")
print(f"h_mpb_from_bank magnitude: {np.abs(h_mpb_from_bank).max():.6e}")
print(f"Max abs diff: {np.max(np.abs(h_mpb_from_wfg - h_mpb_from_bank)):.6e}")
print(
    f"Relative diff: {np.max(np.abs(h_mpb_from_wfg - h_mpb_from_bank)) / np.abs(h_mpb_from_wfg).max():.6e}"
)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# ax1.plot(bank_config["fbin"], h_mpb_from_wfg[0, 0, :].real, label="wfg")
# ax1.plot(bank_config["fbin"], h_mpb_from_bank[0, 0, :].real, label="bank")
# ax1.set_ylabel("Real")
# ax1.legend()

# ax2.plot(bank_config["fbin"], h_mpb_from_wfg[0, 0, :].imag, label="wfg")
# ax2.plot(bank_config["fbin"], h_mpb_from_bank[0, 0, :].imag, label="bank")
# ax2.set_xlabel("f")
# ax2.set_ylabel("Imaginary")
# ax2.legend()

# plt.tight_layout()
# plt.show()

print("\n" + "=" * 80)
print("PREPROCESSING SETUP FOR get_response_over_distance_and_lnlike")
print("=" * 80)

n_phi_incoherent = 64
n_t = 64

batch_intrinsic_indices = np.array([0])
h_impb_batch = amp * np.exp(1j * phase)
sdp.h_impb = h_impb_batch

det_name_inner = sdp.likelihood.event_data.detector_names[0]
tgps_1d = sdp.likelihood.event_data.tgps
lat, lon = skyloc_angles.cart3d_to_latlon(
    skyloc_angles.normalize(DETECTORS[det_name_inner].location)
)

par_dic_transformed = sdp.transform_par_dic_by_sky_poisition(
    det_name_inner, par_dic_0, lon, lat, tgps_1d
)

delay_single = get_geocenter_delays(
    det_name_inner,
    par_dic_transformed["lat"],
    par_dic_transformed["lon"],
)[0]
tcoarse_1d = sdp.likelihood.event_data.tcoarse
dt_sample = sdp.likelihood.event_data.times[1]
t_grid_single = (np.arange(n_t) - n_t // 2) * dt_sample
t_grid_single += par_dic_transformed["t_geocenter"] + tcoarse_1d + delay_single

timeshifts_dbt_single = np.exp(
    -2j * np.pi * t_grid_single[None, None, :] * sdp.likelihood.fbin[None, :, None]
)

print("Setup complete:")
print(f"  h_impb_batch shape: {h_impb_batch.shape}")
print(f"  dh_weights_dmpb shape: {sdp.dh_weights_dmpb.shape}")
print(f"  hh_weights_dmppb shape: {sdp.hh_weights_dmppb.shape}")
print(f"  timeshifts_dbt_single shape: {timeshifts_dbt_single.shape}")
print(f"  t_grid_single shape: {t_grid_single.shape}")
print(f"  lat, lon: {lat:.6f}, {lon:.6f}")
print(f"  delay_single: {delay_single:.6f}")
print(f"  par_dic_transformed[t_geocenter]: {par_dic_transformed['t_geocenter']:.6f}")

print("\n" + "=" * 80)
print("LOOK INTO THE RESPONSE OPTIMIZATION")
print("=" * 80)

# Method 1: Using existing optimized code
print("\nMethod 1: Using get_response_over_distance_and_lnlike")
r_iotp_method1, lnlike_iot_method1 = sdp.get_response_over_distance_and_lnlike(
    sdp.dh_weights_dmpb,
    sdp.hh_weights_dmppb,
    sdp.h_impb,
    timeshifts_dbt_single,
    sdp.likelihood.asd_drift,
    sdp.likelihood_calculator.n_phi,
    sdp.likelihood_calculator.m_arr,
)
print(f"  r_iotp shape: {r_iotp_method1.shape}")
print(f"  lnlike_iot shape: {lnlike_iot_method1.shape}")
max_lnlike = lnlike_iot_method1.max()
max_indices1 = np.unravel_index(np.argmax(lnlike_iot_method1), lnlike_iot_method1.shape)
print(f"  Max lnlike: {max_lnlike:.6f}")
print(f"  Max lnlike indices (i, o, t): {[_.item() for _ in max_indices1]}")
print(f"  r_iotp at max lnlike: {r_iotp_method1[max_indices1]}")

# Method 2: Iterating over t and o explicitly
print("\nMethod 2: Explicit iteration over t and o")

i, m, p, b = h_impb_batch.shape
t = timeshifts_dbt_single.shape[-1]
n_phi = sdp.likelihood_calculator.n_phi

phi_grid_o = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
m_inds, mprime_inds = zip(*itertools.combinations_with_replacement(range(m), 2))
dh_phasor_mo = np.exp(-1j * np.outer(sdp.likelihood_calculator.m_arr, phi_grid_o))
hh_phasor_Mo = np.exp(
    1j
    * np.outer(
        sdp.likelihood_calculator.m_arr[m_inds,]
        - sdp.likelihood_calculator.m_arr[mprime_inds,],
        phi_grid_o,
    )
)

r_iotp_method2 = np.zeros((i, n_phi, t, p))
lnlike_iot_method2 = np.zeros((i, n_phi, t))

for i_idx in range(i):
    for t_idx in range(t):
        timeshift_t = timeshifts_dbt_single[0, :, t_idx]
        for o_idx, phi_o in enumerate(phi_grid_o):
            # Apply timeshift and phase shift
            h_shifted_conj = (
                h_impb_batch[i_idx].conj()
                * timeshift_t[None, :].conj()
                * np.exp(-1j * np.outer(sdp.likelihood_calculator.m_arr, phi_o))[
                    :, None
                ]
            )

            # Compute <d|h> per polarization
            dh_weights = sdp.dh_weights_dmpb[0] * sdp.likelihood.asd_drift[0] ** -2
            dh_p = np.sum(dh_weights * h_shifted_conj, axis=(0, 2)).real

            # Compute <h|h> per polarization pair
            hh_weights = sdp.hh_weights_dmppb[0] * sdp.likelihood.asd_drift[0] ** -2
            hh_pp = np.zeros((p, p))
            for M_idx, (m_idx, mprime_idx) in enumerate(zip(m_inds, mprime_inds)):
                for p_idx in range(p):
                    for pp_idx in range(p):
                        hh_pp[p_idx, pp_idx] += (
                            np.sum(
                                hh_weights[M_idx, p_idx, pp_idx]
                                * h_impb_batch[i_idx, m_idx, p_idx]
                                * h_impb_batch[i_idx, mprime_idx, pp_idx].conj()
                            )
                            * hh_phasor_Mo[M_idx, o_idx]
                        ).real

            # Invert matrix and solve for r
            hh_det = hh_pp[0, 0] * hh_pp[1, 1] - hh_pp[0, 1] * hh_pp[1, 0]
            if i_idx == 0 and t_idx == 0 and o_idx == 0:
                print(f"DEBUG: hh_pp = {hh_pp}")
                print(f"DEBUG: hh_det = {hh_det}")
                print(f"DEBUG: dh_p = {dh_p}")
            if hh_det > 0:
                hh_inv = (
                    np.array([[hh_pp[1, 1], -hh_pp[0, 1]], [-hh_pp[1, 0], hh_pp[0, 0]]])
                    / hh_det
                )
                r_p = hh_inv @ dh_p
            else:
                r_p = np.zeros(p)

            r_iotp_method2[i_idx, o_idx, t_idx] = r_p

            # Compute lnlike
            lnlike_iot_method2[i_idx, o_idx, t_idx] = np.sum(r_p * dh_p) - 0.5 * np.sum(
                r_p[:, None] * r_p[None, :] * hh_pp
            )

print(f"  r_iotp shape: {r_iotp_method2.shape}")
print(f"  lnlike_iot shape: {lnlike_iot_method2.shape}")
print(f"  Max lnlike: {lnlike_iot_method2.max():.6f}")
max_indices2 = np.unravel_index(np.argmax(lnlike_iot_method2), lnlike_iot_method2.shape)
print(f"  Max lnlike indices (i, o, t) (Method 2): {[_.item() for _ in max_indices2]}")
print(f"  r_iotp at max lnlike (Method 2): {r_iotp_method2[max_indices2]}")

# Compare results
print("\nComparison:")
print(
    f"  Max lnlike difference: {np.abs(lnlike_iot_method1 - lnlike_iot_method2).max():.6e}"
)
print(f"  Max r difference: {np.abs(r_iotp_method1 - r_iotp_method2).max():.6e}")
print(
    f"  Relative r difference: {np.abs(r_iotp_method1 - r_iotp_method2).max() / np.abs(r_iotp_method1).max():.6e}"
)

print("\n" + "=" * 80)
print("COMMON LIKELIHOOD EVALUATION")
print("=" * 80)

# Get max indices for Method 1
i_idx1, o_idx1, t_idx1 = max_indices1

# Reconstruct parameter dictionary
intrinsic_params_from_bank = intrinsic_bank_df.iloc[
    batch_intrinsic_indices[i_idx1]
].to_dict()
intrinsic_params_from_bank["f_ref"] = bank_config["f_ref"]
intrinsic_params_from_bank["l1"] = 0
intrinsic_params_from_bank["l2"] = 0

psi_recon, d_luminosity_recon = sdp.bestfit_response_to_psi_and_d_luminosity(
    r_iotp_method1[max_indices1], sdp.likelihood.event_data.detector_names[0]
)

phi_ref_recon = o_idx1 / n_phi * 2 * np.pi

dt_linfree_for_this_sample = sdp.intrinsic_sample_processor.cached_dt_linfree_relative[
    batch_intrinsic_indices[i_idx1]
]
t_geocenter_recon = (
    par_dic_transformed["t_geocenter"]
    + t_grid_single[t_idx1]
    - tcoarse_1d
    - delay_single
    + dt_linfree_for_this_sample
)

reconstructed_par_dic = intrinsic_params_from_bank | {
    "ra": par_dic_transformed["ra"],
    "dec": par_dic_transformed["dec"],
    "t_geocenter": t_geocenter_recon,
    "phi_ref": phi_ref_recon,
    "psi": psi_recon,
    "d_luminosity": d_luminosity_recon,
}

print("Reconstructed parameters:")
print(f"  psi: {psi_recon:.6f}")
print(f"  d_luminosity: {d_luminosity_recon:.6f}")
print(f"  phi_ref: {phi_ref_recon:.6f}")
print(f"  t_geocenter: {t_geocenter_recon:.6f}")

# Compute likelihood using coherent_posterior
lnlike_from_coherent = coherent_posterior.likelihood.lnlike(reconstructed_par_dic)

print("\nLikelihood comparison:")
print(f"  From barebones calculation: {max_lnlike:.6f}")
print(f"  From coherent_posterior.lnlike: {lnlike_from_coherent:.6f}")
print(f"  Difference: {max_lnlike - lnlike_from_coherent:.6f}")

if abs(max_lnlike - lnlike_from_coherent) < 0.01:
    print("\n[PASS] Likelihoods match! Traceability verified.")
else:
    print("\n[FAIL] Likelihoods don't match. Need to investigate further.")

print("\n" + "=" * 80)
print("BAREBONE LIKELIHOOD EVALUATION")
print("=" * 80)

# Step-by-step barebone implementation tracing through waveform generation,
# time shifts, detector responses, and relative binning weights
likelihood = coherent_posterior.likelihood
wfg_coherent = likelihood.waveform_generator
fbin_coherent = likelihood.fbin
event_data_coherent = likelihood.event_data

print("\nStep 1: Generate waveform h_mpb from parameters (by mode)")
# Get intrinsic parameters (excluding extrinsic ones that will be applied separately)
intrinsic_pars = {
    k: v
    for k, v in reconstructed_par_dic.items()
    if k not in ["ra", "dec", "t_geocenter", "phi_ref", "psi", "d_luminosity"]
}
# Use default extrinsic params for waveform generation
waveform_par_dic = intrinsic_pars | config.DEFAULT_PARAMS_DICT
h_mpb_barebone = wfg_coherent.get_hplus_hcross(
    fbin_coherent, waveform_par_dic, by_m=True
)
print(f"  h_mpb_barebone shape: {h_mpb_barebone.shape} (m, p, b)")
print(f"  h_mpb_barebone dtype: {h_mpb_barebone.dtype}")
print(f"  h_mpb_barebone sample (m=0, p=0, b=0): {h_mpb_barebone[0, 0, 0]:.6e}")
# Apply linear-free time shift to match bank waveform
dt_linfree_barebone = sdp.intrinsic_sample_processor.cached_dt_linfree_relative[
    batch_intrinsic_indices[0]
]
timeshift_linfree = np.exp(2j * np.pi * dt_linfree_barebone * fbin_coherent)
h_mpb_barebone_with_linfree = h_mpb_barebone * timeshift_linfree[None, None, :]
print(f"  dt_linfree_barebone: {dt_linfree_barebone:.6e}")
print(
    f"  h_mpb_barebone_with_linfree sample (m=0, p=0, b=0): {h_mpb_barebone_with_linfree[0, 0, 0]:.6e}"
)
# Comparison: Compare with h_impb_batch from single-detector code (bank waveform)
h_impb_single = h_impb_batch[0]  # (m, p, b) - first intrinsic sample from bank
print("\n  COMPARISON with single-detector h_impb (from bank):")
print(f"    h_impb_single shape: {h_impb_single.shape}")
print(f"    h_impb_single sample (m=0, p=0, b=0): {h_impb_single[0, 0, 0]:.6e}")
print(
    f"    Max abs diff: {np.max(np.abs(h_mpb_barebone_with_linfree - h_impb_single)):.6e}"
)
print(
    f"    Relative diff: {np.max(np.abs(h_mpb_barebone_with_linfree - h_impb_single)) / np.max(np.abs(h_mpb_barebone_with_linfree)):.6e}"
)

print("\nStep 2: Apply orbital phase shift (phi_ref)")
m_arr_coherent = (
    wfg_coherent._harmonic_modes_by_m
)  # Get m values from waveform generator
phi_ref = reconstructed_par_dic["phi_ref"]
phi_shift = np.exp(
    +1j * np.array([m for m in m_arr_coherent.keys()])[:, None, None] * phi_ref
)
# Apply phi_ref shift to the waveform with linear-free time shift already applied
h_mpb_phased = h_mpb_barebone_with_linfree * phi_shift
print(f"  phi_ref: {phi_ref:.6f}")
print(f"  phi_shift shape: {phi_shift.shape}")
print(f"  h_mpb_phased sample (m=0, p=0, b=0): {h_mpb_phased[0, 0, 0]:.6e}")
# Comparison: Apply same phi_ref shift to single-detector waveform
i_idx1, o_idx1, t_idx1 = max_indices1
phi_ref_single = o_idx1 / n_phi * 2 * np.pi
phi_shift_single = np.exp(
    +1j * np.outer(sdp.likelihood_calculator.m_arr, [phi_ref_single])
)[:, :, None]
h_impb_phased_single = h_impb_single * phi_shift_single
print("\n  COMPARISON with single-detector after phi_ref shift:")
print(f"    phi_ref_single: {phi_ref_single:.6f}")
print(
    f"    h_impb_phased_single sample (m=0, p=0, b=0): {h_impb_phased_single[0, 0, 0]:.6e}"
)
print(f"    Max abs diff: {np.max(np.abs(h_mpb_phased - h_impb_phased_single)):.6e}")
print(
    f"    Relative diff: {np.max(np.abs(h_mpb_phased - h_impb_phased_single)) / np.max(np.abs(h_mpb_phased)):.6e}"
)

print("\nStep 3: Apply distance scaling (1/d_luminosity)")
d_luminosity = reconstructed_par_dic["d_luminosity"]
h_mpb_scaled = h_mpb_phased / d_luminosity
print(f"  d_luminosity: {d_luminosity:.6f}")
print(f"  h_mpb_scaled sample (m=0, p=0, b=0): {h_mpb_scaled[0, 0, 0]:.6e}")
# Comparison: Apply same distance scaling to single-detector waveform
h_impb_scaled_single = h_impb_phased_single / d_luminosity
print("\n  COMPARISON with single-detector after distance scaling:")
print(
    f"    h_impb_scaled_single sample (m=0, p=0, b=0): {h_impb_scaled_single[0, 0, 0]:.6e}"
)
print(f"    Max abs diff: {np.max(np.abs(h_mpb_scaled - h_impb_scaled_single)):.6e}")
print(
    f"    Relative diff: {np.max(np.abs(h_mpb_scaled - h_impb_scaled_single)) / np.max(np.abs(h_mpb_scaled)):.6e}"
)

print("\nStep 4: Time shift analysis")
print("=" * 80)
print("SINGLE-DETECTOR METHOD: Time shifts applied to pure waveform")
print("=" * 80)
print("Starting from pure waveform (no time shifts applied yet):")
print(
    f"  1. Linear-free time shift: {dt_linfree_barebone:.6f} (already in bank waveform)"
)
print(f"  2. Base time components:")
print(
    f"     - par_dic_transformed['t_geocenter']: {par_dic_transformed['t_geocenter']:.6f}"
)
print(f"     - delay_single (geocenter delay): {delay_single:.6f}")
print(f"     - tcoarse: {tcoarse_1d:.6f}")
print(
    f"     - Base time = t_geocenter + delay + tcoarse = {par_dic_transformed['t_geocenter'] + delay_single + tcoarse_1d:.6f}"
)
t_grid_offset = t_grid_single[t_idx1] - (
    par_dic_transformed["t_geocenter"] + delay_single + tcoarse_1d
)
print(f"  3. Time grid offset (from optimization): {t_grid_offset:.6f}")
print(
    f"  4. TOTAL time applied in single-detector: t_grid_single[{t_idx1}] = {t_grid_single[t_idx1]:.6f}"
)
print()
print("COHERENT LIKELIHOOD METHOD: What t_geocenter should be")
print("=" * 80)
# Convert RA/DEC to lat/lon for get_geocenter_delays
ra = reconstructed_par_dic["ra"]
dec = reconstructed_par_dic["dec"]
gmst = GreenwichMeanSiderealTime(event_data_coherent.tgps)
lat_coherent = dec
lon_coherent = skyloc_angles.ra_to_lon(ra, gmst)
detector_names_coherent = event_data_coherent.detector_names
geocenter_delays = get_geocenter_delays(
    detector_names_coherent, lat_coherent, lon_coherent
)
tcoarse_coherent = event_data_coherent.tcoarse
print(
    "For coherent likelihood, the time shift is: t_geocenter + tcoarse + geocenter_delay"
)
print(
    f"  Current reconstructed t_geocenter: {reconstructed_par_dic['t_geocenter']:.6f}"
)
print(f"  geocenter_delay: {geocenter_delays[0]:.6f}")
print(f"  tcoarse: {tcoarse_coherent:.6f}")
print(
    f"  Total time = {reconstructed_par_dic['t_geocenter'] + tcoarse_coherent + geocenter_delays[0]:.6f}"
)
print()
print("RELATIONSHIP:")
print("=" * 80)
print("To match single-detector, coherent should use:")
print(f"  t_geocenter + tcoarse + geocenter_delay = t_grid_single[{t_idx1}]")
print(f"  t_geocenter = t_grid_single[{t_idx1}] - tcoarse - geocenter_delay")
print(
    f"  t_geocenter = {t_grid_single[t_idx1]:.6f} - {tcoarse_coherent:.6f} - {geocenter_delays[0]:.6f}"
)
t_geocenter_expected = t_grid_single[t_idx1] - tcoarse_coherent - geocenter_delays[0]
print(f"  t_geocenter_expected = {t_geocenter_expected:.6f}")
print(
    f"  Current t_geocenter_reconstructed = {reconstructed_par_dic['t_geocenter']:.6f}"
)
print(
    f"  Difference = {reconstructed_par_dic['t_geocenter'] - t_geocenter_expected:.6f}"
)
print()
# Compute timeshifts for comparison
timeshift_single_at_t = timeshifts_dbt_single[0, :, t_idx1]  # (b,)
total_delays_coherent = (
    reconstructed_par_dic["t_geocenter"] + tcoarse_coherent + geocenter_delays
)
timeshifts_dbf = np.exp(
    -2j * np.pi * total_delays_coherent[:, None] * fbin_coherent[None, :]
)
timeshift_expected = np.exp(-2j * np.pi * t_grid_single[t_idx1] * fbin_coherent)

print("\nStep 5: Apply time shifts to waveform")
# Use the expected time shift that matches single-detector
h_dmpb_timeshifted = (
    h_mpb_scaled[None, :, :, :] * timeshift_expected[None, None, None, :]
)
print(f"  h_dmpb_timeshifted shape: {h_dmpb_timeshifted.shape} (d, m, p, b)")
print(
    f"  h_dmpb_timeshifted sample (d=0, m=0, p=0, b=0): {h_dmpb_timeshifted[0, 0, 0, 0]:.6e}"
)
# Comparison: Apply same time shift to single-detector waveform
h_impb_timeshifted_single = h_impb_scaled_single * timeshift_single_at_t[None, None, :]
# print("\n  COMPARISON with single-detector after time shift:")
# print(f"    h_impb_timeshifted_single shape: {h_impb_timeshifted_single.shape}")
# print(
#     f"    h_impb_timeshifted_single sample (m=0, p=0, b=0): {h_impb_timeshifted_single[0, 0, 0]:.6e}"
# )
# print(
#     f"    Max abs diff: {np.max(np.abs(h_dmpb_timeshifted[0] - h_impb_timeshifted_single)):.6e}"
# )
# print(
#     f"    Relative diff: {np.max(np.abs(h_dmpb_timeshifted[0] - h_impb_timeshifted_single)) / np.max(np.abs(h_dmpb_timeshifted[0])):.6e}"
# )

# print("\nStep 6: Apply detector responses (fplus, fcross)")
lat_coherent = dec  # In cogwheel, dec is used as lat
gmst = GreenwichMeanSiderealTime(event_data_coherent.tgps)
lon_coherent = skyloc_angles.ra_to_lon(ra, gmst)
psi = reconstructed_par_dic["psi"]
fplus_fcross_0 = get_fplus_fcross_0(detector_names_coherent, lat_coherent, lon_coherent)
# fplus_fcross_0 shape: (d, P) where d=detectors, P=2 (plus, cross)
# Apply psi rotation
psi_rot = np.array(
    [[np.cos(2 * psi), np.sin(2 * psi)], [-np.sin(2 * psi), np.cos(2 * psi)]]
)  # (P, p)
response_dp = fplus_fcross_0 @ psi_rot  # (d, p)
response_p = response_dp[0]  # (p,) - extract first detector
# Apply response: h_dmpb -> h_dmb with response applied
# h_dmpb_timeshifted is (d, m, p, b), response_p is (p,)
h_dmb_responded = np.einsum("mpb,p->mb", h_dmpb_timeshifted[0], response_p)[
    None, :, :
]  # (1, m, b)
# print(f"  response_dp shape: {response_dp.shape} (d, p)")
# print(f"  response_dp sample (d=0, p=0): {response_dp[0, 0]:.6e}")
# print(f"  h_dmb_responded shape: {h_dmb_responded.shape} (d, m, b)")
# print(f"  h_dmb_responded sample (d=0, m=0, b=0): {h_dmb_responded[0, 0, 0]:.6e}")
# Comparison: Apply same detector response to single-detector waveform
# Get response from single-detector (it's at detector zenith, so response should match)
r_max = r_iotp_method1[max_indices1]
psi_single, d_lum_single = sdp.bestfit_response_to_psi_and_d_luminosity(
    r_max, det_name_inner
)
# Compute response for single-detector (should be same as coherent since sky position is fixed)
response_single = sdp.psi_and_d_luminosity_to_response(
    psi_single, d_lum_single, det_name_inner
).squeeze()
h_impb_responded_single = np.einsum(
    "mpb,p->mb", h_impb_timeshifted_single, response_single
)
# print("\n  COMPARISON with single-detector after detector response:")
# print(f"    response_single: {response_single}")
# print(f"    h_impb_responded_single shape: {h_impb_responded_single.shape}")
# print(
#     f"    h_impb_responded_single sample (m=0, b=0): {h_impb_responded_single[0, 0]:.6e}"
# )
# print(
#     f"    Max abs diff: {np.max(np.abs(h_dmb_responded[0] - h_impb_responded_single)):.6e}"
# )
# print(
#     f"    Relative diff: {np.max(np.abs(h_dmb_responded[0] - h_impb_responded_single)) / np.max(np.abs(h_dmb_responded[0])):.6e}"
# )

# print("\nStep 7: Get relative binning weights and compute <d|h> and <h|h>")
# For now, use the likelihood's internal methods but we've traced the transformations above
# The weights are computed from a reference waveform and stored on the likelihood
h_f_barebone = coherent_posterior.likelihood._get_h_f(reconstructed_par_dic)
d_h_barebone = coherent_posterior.likelihood._compute_d_h(h_f_barebone)
h_h_barebone = coherent_posterior.likelihood._compute_h_h(h_f_barebone)
# print(f"  d_h_barebone shape: {d_h_barebone.shape}")
# print(f"  d_h_barebone per detector: {d_h_barebone}")
# print(f"  h_h_barebone shape: {h_h_barebone.shape}")
# print(f"  h_h_barebone per detector: {h_h_barebone}")

# print("\nStep 8: Sum over detectors and compute log-likelihood")
d_h_total = np.sum(d_h_barebone.real)
h_h_total = np.sum(h_h_barebone.real)
# print(f"  d_h_total: {d_h_total:.6e}")
# print(f"  h_h_total: {h_h_total:.6e}")

# Compute log-likelihood (distance-fitted version)
lnlike_from_barebone = 0.5 * (d_h_total**2) / h_h_total * (d_h_total > 0)

# print("\nBarebone likelihood comparison:")
# print(f"  From coherent_posterior.lnlike: {lnlike_from_coherent:.6f}")
# print(f"  From barebone step-by-step: {lnlike_from_barebone:.6f}")
# print(f"  Difference: {abs(lnlike_from_coherent - lnlike_from_barebone):.6e}")

# if abs(lnlike_from_coherent - lnlike_from_barebone) < 1e-10:
#     print("\n[PASS] Barebone implementation matches coherent_posterior.lnlike()!")
# else:
#     print(
#         "\n[WARNING] Barebone implementation does not exactly match coherent_posterior.lnlike()"
#     )
#     print(
#         "  This may indicate a subtle difference in implementation or numerical precision."
#     )


## 1-d time scan
t_scan = np.linspace(-0.1, +0.1, 2048)
lnlike_t_scan = np.array(
    [
        coherent_posterior.likelihood.lnlike(reconstructed_par_dic | {"t_geocenter": t})
        for t in t_scan
    ]
)

# print(f"{lnlike_t_scan.max()=}")
# print(f"{t_scan[lnlike_t_scan.argmax()]=}")


# print(f"{t_scan[lnlike_t_scan.argmax()]=}")
# print(f"{sdp.intrinsic_sample_processor.cached_dt_linfree_relative[0]=}")
# print(f"{delay_single=}")
# print(f"{par_dic_transformed['t_geocenter']=}")

print("\n" + "=" * 80)
print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
