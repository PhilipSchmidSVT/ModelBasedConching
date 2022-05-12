"Library handling data collection from excel sheets."

from __future__ import annotations
from functools import reduce

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ConchingModel.Data.Timeseries import (
    PhaseLabel,
    SubstanceLabel,
    aSample,
    aTimeseries,
    aPhase,
)


@dataclass
class aRequest:
    """
    Handling of specific data requests for extraction.
    Example:
    request = aRequest(SubstanceLabel.Benzaldehyd, PhaseLabel.COCOABUTTER, "KB", 'data\\farinograph\\2022_04_27_farinograph\\Auswertung_KB-Luft_50rpm_Wiederholung_YvGu - Copy.xlsx', 20, 'Ansatz 20 min')
    """

    substance: SubstanceLabel
    phase: PhaseLabel
    sheet: str
    data_src: str
    minutes: int
    category: str

    @property
    def time(self) -> float:
        "Time in minutes to time in hour fractions."
        return self.minutes / 60

    def _get_multiindex(self, df: pd.DataFrame) -> pd.MultiIndex:
        "Create the multiindex to find data by."
        idx_substance = list(
            filter(
                lambda x: not x.startswith("Ansatz") and x != "Einwaage [g]",
                df.index.dropna(),
            )
        )
        idx_substance_unique = [
            id_sub
            for id_sub in idx_substance
            if id_sub
            in (
                "Essigsäure",
                "TMP",
                "Linalool",
                "Benzaldehyd",
                "Phenylethanol",
                "Phenylethylacetat",
            )
        ]
        idx_substance_unique = list(dict.fromkeys(idx_substance_unique))
        idx_time = list(filter(lambda x: "Ansatz" in x, df.index.dropna()))
        multi_index = pd.MultiIndex.from_tuples(
            tuple(itertools.product(idx_time, idx_substance_unique))
        )
        return multi_index

    def _clean_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        "Drop all nan indexed rows and infer the right types of each column."
        df = df[df.index.notnull()].dropna()
        df = df.infer_objects()
        return df

    def get_sample(self) -> aSample:
        "Get a sample from a given request and turn it into aSample."
        return aSample(
            self.phase.name,
            self.minutes,
            [
                self.value,
            ],
        )

    @property
    def df(self) -> pd.DataFrame:
        "Get dataframe and respective sheet."
        df = pd.read_excel(
            self.data_src,
            self.sheet,
            index_col=0,
            header=0,
            usecols="A,I",
            dtype={"a": str, "i": np.float64},
        ).dropna(axis=1, how="all")
        multi_index = self._get_multiindex(df)
        df = self._clean_frame(df)
        df.index = multi_index
        return df

    @property
    def value(self) -> float:
        "Return the associated value."
        return self.df.loc[self.category, self.substance.name].values[0]

    @staticmethod
    def new(
        substance: SubstanceLabel,
        phase: PhaseLabel,
        sheet: str,
        data_src: str,
        minutes: int,
        category: str,
    ) -> aRequest:
        "Construct a new instance of request."
        return aRequest(substance, phase, sheet, data_src, minutes, category)


@dataclass
class Extractor:
    "Handles the logic that combines a number of requests."
    requests: list[aRequest]

    def add_request(self, request: aRequest) -> Extractor:
        "Add a request to the extractor."
        return Extractor(self.requests.append(request))

    def _convert_minutes_to_float_time(
        self, samples: list[aSample], minutes: int
    ) -> list[aSample]:
        samples = [sample for sample in samples if sample.sampling_time == minutes]
        samples = [
            aSample(sample.label, sample.sampling_time / 60, sample.concentration)
            for sample in samples
        ]
        return samples

    def _group_requests(self) -> dict:
        "Group requests by a dict according to their sampling time."
        assert len({request.phase for request in self.requests}) == 1
        assert len({request.substance for request in self.requests}) == 1
        samples = [request.get_sample() for request in self.requests]
        times_unique = {sample.sampling_time for sample in samples}
        result = []
        for itime in times_unique:
            isamples = self._convert_minutes_to_float_time(samples, itime)
            sample = reduce(lambda x, y: x.concat(y), isamples)
            result.append(sample)
        return result

    def evaluate_requests(self) -> aTimeseries:
        "Get the according value of each request."
        samples = self._group_requests()
        sampling_times = sorted(list({sample.sampling_time for sample in samples}))
        phase_label = samples[0].label
        phase = aPhase(phase_label, sampling_times, sorted(samples))
        ts = aTimeseries(
            [
                phase,
            ],
            sampling_times,
        )
        return ts

    @staticmethod
    def new() -> Extractor:
        "Construct a new Extractor."
        return Extractor([])
