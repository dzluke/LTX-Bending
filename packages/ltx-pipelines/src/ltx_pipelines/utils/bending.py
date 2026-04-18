from typing import Callable

import torch
from tqdm import tqdm

from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.model.transformer import X0Model
from ltx_pipelines.utils.samplers import _step_state
from ltx_pipelines.utils.types import Denoiser, LatentState

VideoBendingFn = Callable[[LatentState, int], LatentState]


def make_network_bending_loop(
    bend_video: VideoBendingFn,
) -> Callable[
    [torch.Tensor, LatentState | None, LatentState | None, DiffusionStepProtocol, X0Model, Denoiser],
    tuple[LatentState | None, LatentState | None],
]:
    """Build an Euler denoising loop that applies ``bend_video`` to the video state at every step."""

    def loop(
        sigmas: torch.Tensor,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        stepper: DiffusionStepProtocol,
        transformer: X0Model,
        denoiser: Denoiser,
    ) -> tuple[LatentState | None, LatentState | None]:
        for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
            if video_state is not None:
                video_state = bend_video(video_state, step_idx)

            denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_idx)

            video_state = _step_state(video_state, denoised_video, stepper, sigmas, step_idx)
            audio_state = _step_state(audio_state, denoised_audio, stepper, sigmas, step_idx)

        return (video_state, audio_state)

    return loop
