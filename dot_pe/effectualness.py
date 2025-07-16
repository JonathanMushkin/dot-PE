import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from . import waveform_overlap
from . import evidence_calculator
from . import sample_generation
from .sampler_free_utils import clear_cache


def calculate_effectualness(
    bank_folder,
    block_size=32,
    device="cpu",
    n_phi=32,
    n_t=32,
    save_every=16,
    n_eff=None,
    n_bank=None,
    eff_folder=None,
):
    bank_folder = Path(bank_folder)
    if eff_folder is None:
        eff_folder = bank_folder / "effectualness"
    else:
        eff_folder = Path(eff_folder)

    with open(bank_folder / "bank_config.json", "r") as fp:
        bank_config = json.load(fp)
    n_bank = n_bank if n_bank else bank_config["bank_size"]

    extrinsic_samples = pd.read_feather(eff_folder / "extrinsic_samples.feather")
    n_eff = n_eff if n_eff else extrinsic_samples.shape[0]
    extrinsic_samples = extrinsic_samples.iloc[:n_eff]
    isp_e = waveform_overlap.get_intrinsic_sample_processor(
        bank_folder=eff_folder,
    )

    isp_b = waveform_overlap.get_intrinsic_sample_processor(bank_folder=bank_folder)

    esp = evidence_calculator.ExtrinsicSampleProcessor("L")
    fbin = np.asarray(bank_config["fbin"])
    h_h_weights_dmmppb = torch.as_tensor(
        waveform_overlap.get_weights(bank_folder), device=device
    )

    amp, phase = isp_e.load_amp_and_phase(
        isp_e.waveform_dir, indices=np.arange(n_eff, dtype=int)
    )

    response_dpe, _ = esp.get_components(
        extrinsic_samples, fbin, 0
    )  # ignore times shifts

    response_dpe = torch.as_tensor(response_dpe, device=device)

    phasor = torch.exp(
        1j
        * torch.as_tensor(
            extrinsic_samples["phi_ref"].values[:, None],
            device=device,
        )
        * torch.as_tensor(
            isp_b.likelihood.waveform_generator.m_arr[None, :],
            device=device,
        ),
    )
    h_eff_impb = torch.as_tensor(amp * np.exp(1j * phase), device=device)
    h_eff_impb = torch.einsum(
        "impb, dpi, im->impb",
        h_eff_impb,
        response_dpe,
        phasor,
    )

    h_eff_norm_i = torch.einsum(
        "dmMpPb, impb, iMPb->i",
        h_h_weights_dmmppb,
        h_eff_impb,
        h_eff_impb.conj(),
    ).real

    h_eff_impb = h_eff_impb / torch.sqrt(h_eff_norm_i.reshape(-1, 1, 1, 1))
    del h_eff_norm_i, response_dpe, phasor, amp, phase
    clear_cache(device)

    t_arr = (
        torch.arange(n_t, device=device) - n_t // 2
    ) * isp_b.likelihood.event_data.times[1]
    phi_arr = torch.linspace(
        start=0.0, end=torch.pi * 2, steps=n_phi + 1, device=device
    )[:-1]

    n_blocks = -(n_bank // -block_size)
    overlaps_ij = np.zeros((n_eff, n_bank), dtype=np.float16)

    print(f"starting {n_blocks} blocks")
    for b in range(n_blocks):
        print(f"starting block {b + 1}/{n_blocks}")

        inds = np.arange(b * block_size, min((b + 1) * block_size, n_bank))
        overlaps_ij[:, inds] = (
            _overlap_with_bank_block(
                inds,
                isp_b,
                h_eff_impb,
                t_arr,
                phi_arr,
                bank_folder,
                h_h_weights_dmmppb,
                device,
            )
            .cpu()
            .numpy()
        )

        if (b + 1) % save_every == 0:
            save_path = eff_folder / f"overlaps_ij_{b + 1:04d}.pt"
            np.save(arr=overlaps_ij, file=save_path)
            clear_cache(device)
            print(f"Saving! to {save_path}")

    save_path = eff_folder / "overlaps_ij.npy"
    np.save(arr=overlaps_ij, file=save_path)
    del overlaps_ij
    clear_cache(device)


def _overlap_with_bank_block(
    inds,
    isp_b,
    h_eff_impb,
    t_arr,
    phi_arr,
    bank_folder,
    h_h_weights_dmmppb,
    device,
):
    n_t = len(t_arr)
    amp, phase = isp_b.load_amp_and_phase(isp_b.waveform_dir, indices=inds)

    h_bank_jtompb = torch.as_tensor(
        waveform_overlap.apply_time_and_phase_shift(
            bank_folder,
            amp * np.exp(1j * phase),
            t_arr,
            phi_arr,
        ),
        device=device,
    )
    hh_bank_jopp = torch.einsum(
        "dmMpPb, jompb, joMPb->jopP",
        h_h_weights_dmmppb,
        h_bank_jtompb[:, n_t // 2, ...],
        h_bank_jtompb[:, n_t // 2, ...].conj(),
    ).real
    hh_eff_bank_ijtop = torch.einsum(
        "dmMpPb, impb, jtoMPb->ijtoP",
        h_h_weights_dmmppb,
        h_eff_impb,
        h_bank_jtompb.conj(),
    ).real

    del h_bank_jtompb
    B_inv = torch.linalg.inv(hh_bank_jopp)
    del hh_bank_jopp

    denom = torch.einsum(
        "ijtop, jopP, ijtoP->ijto",
        hh_eff_bank_ijtop,
        B_inv,
        hh_eff_bank_ijtop,
    )

    bestfit_overlap_ijto = torch.sqrt(denom)
    overlap_ij = bestfit_overlap_ijto.amax(dim=(2, 3))
    return overlap_ij


def create_effectualness_test_set(bank_folder, n_eff, eff_folder=None):
    """
    Create a test set for the effectualness calculation.
    """
    bank_folder = Path(bank_folder)
    if eff_folder is None:
        eff_folder = bank_folder / "effectualness"
    else:
        eff_folder = Path(eff_folder)

    with open(bank_folder / "bank_config.json", "r") as fp:
        bank_config = json.load(fp)
    # backward compatibility
    inc_faceon_factor = (
        bank_config["inc_faceon_factor"] if "inc_faceon_factor" in bank_config else 2
    )
    sample_generation.main(
        bank_size=n_eff,
        q_min=bank_config["q_min"],
        m_min=bank_config["min_mchirp"],
        m_max=bank_config["max_mchirp"],
        inc_faceon_factor=inc_faceon_factor,
        f_ref=bank_config["f_ref"],
        fbin=np.array(bank_config["fbin"]),
        blocksize=n_eff,
        bank_dir=eff_folder,
        approximant=bank_config["approximant"],
    )

    tgps = 0
    psi = np.random.uniform(0, 2 * np.pi, n_eff)
    sin_lat = np.random.uniform(-1, 1, n_eff)
    lon = np.random.uniform(0, 2 * np.pi, n_eff)
    phi_ref = np.random.uniform(0, 2 * np.pi, n_eff)
    t_geocenter = np.zeros(n_eff)
    d_luminosity = np.ones(n_eff) * 1.0

    extrinsic_samples = pd.DataFrame(
        {
            "tgps": tgps,
            "psi": psi,
            "lat": np.arcsin(sin_lat),
            "lon": lon,
            "phi_ref": phi_ref,
            "t_geocenter": t_geocenter,
            "d_luminosity": d_luminosity,
        }
    )
    filename = eff_folder / "extrinsic_samples.feather"
    extrinsic_samples.to_feather(filename)


def create_effectualness_test_set_from_physical_prior(
    bank_folder, n_eff, eff_folder=None
):
    """
    Create a test set for the effectualness calculation.
    """
    bank_folder = Path(bank_folder)
    if eff_folder is None:
        eff_folder = bank_folder / "effectualness"
    else:
        eff_folder = Path(eff_folder)

    with open(bank_folder / "bank_config.json", "r") as fp:
        bank_config = json.load(fp)
    sample_generation.create_physical_prior_bank(
        bank_size=n_eff,
        q_min=bank_config["q_min"],
        m_min=bank_config["min_mchirp"],
        m_max=bank_config["max_mchirp"],
        f_ref=bank_config["f_ref"],
        fbin=np.array(bank_config["fbin"]),
        blocksize=n_eff,
        bank_dir=eff_folder,
        approximant=bank_config["approximant"],
    )

    tgps = 0
    psi = np.random.uniform(0, 2 * np.pi, n_eff)
    sin_lat = np.random.uniform(-1, 1, n_eff)
    lon = np.random.uniform(0, 2 * np.pi, n_eff)
    phi_ref = np.random.uniform(0, 2 * np.pi, n_eff)
    t_geocenter = np.zeros(n_eff)
    d_luminosity = np.ones(n_eff) * 1.0

    extrinsic_samples = pd.DataFrame(
        {
            "tgps": tgps,
            "psi": psi,
            "lat": np.arcsin(sin_lat),
            "lon": lon,
            "phi_ref": phi_ref,
            "t_geocenter": t_geocenter,
            "d_luminosity": d_luminosity,
        }
    )
    filename = eff_folder / "extrinsic_samples.feather"
    extrinsic_samples.to_feather(filename)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate effectualness of a bank.")
    parser.add_argument("--bank_folder", type=str, help="Path to the bank folder.")
    parser.add_argument(
        "--eff_folder",
        type=str,
        help="Path to the effectualness folder.",
        default=None,
    )
    parser.add_argument("--block_size", type=int, default=32, help="Block size.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use.")
    parser.add_argument("--n_phi", type=int, default=32, help="Number of phi samples.")
    parser.add_argument("--n_t", type=int, default=32, help="Number of time samples.")
    parser.add_argument(
        "--save_every", type=int, default=16, help="Save every N blocks."
    )
    parser.add_argument(
        "--n_eff",
        type=int,
        default=None,
        help="Number of effectualness samples.",
    )
    parser.add_argument(
        "--n_bank", type=int, default=None, help="Number of bank samples."
    )

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_arguments()
    eff_folder = args.pop("eff_folder", None)
    if eff_folder is None:
        eff_folder = Path(args["bank_folder"]) / "effectualness"
    else:
        eff_folder = Path(eff_folder)
    eff_folder.mkdir(parents=True, exist_ok=True)
    if not (eff_folder / "extrinsic_samples.feather").exists():
        create_effectualness_test_set_from_physical_prior(
            args["bank_folder"], args["n_eff"], eff_folder
        )
        # Optionally call the function if needed
        # create_effectualness_test_set_from_physical_prior(args["bank_folder"], args["n_eff"], eff_folder)
    args |= {"eff_folder": eff_folder}
    calculate_effectualness(**args)
