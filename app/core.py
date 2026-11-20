"""Synthetic field histories, pattern mining, and next-item evaluation."""
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import get_close_matches
import random


@dataclass(frozen=True)
class Item:
    kind: str
    description: str
    next_step: str


ITEMS = {
    "wheat": Item("crop", "Wheat was recorded for the season.", "Review the recorded rotation history."),
    "rice": Item("crop", "Rice was recorded for the season.", "Confirm the crop entry in the field log."),
    "maize": Item("crop", "Maize was recorded for the season.", "Review the recorded rotation history."),
    "soybean": Item("crop", "Soybean was recorded for the season.", "Confirm the crop entry in the field log."),
    "cotton": Item("crop", "Cotton was recorded for the season.", "Confirm the crop entry in the field log."),
    "chickpea": Item("crop", "Chickpea was recorded for the season.", "Review the recorded rotation history."),
    "aphids_observed": Item("observation", "Aphids were recorded during scouting.", "Document the observation for a qualified adviser."),
    "leaf_rust": Item("observation", "Rust-like symptoms were recorded.", "Record the symptoms and seek qualified identification."),
    "stem_borer": Item("observation", "Borer-like damage was recorded.", "Log the location and seek qualified identification."),
    "boll_damage": Item("observation", "Damage to cotton bolls was recorded.", "Record the symptoms and seek qualified identification."),
    "drought_stress": Item("observation", "Water-stress symptoms were recorded.", "Check that the field observations were entered correctly."),
    "humid_season": Item("observation", "The synthetic season was humid.", "Review the weather and scouting entries."),
    "dry_season": Item("observation", "The synthetic season was dry.", "Review the weather and scouting entries."),
    "mild_season": Item("observation", "The synthetic season was mild.", "Review the weather and scouting entries."),
    "fungicide_applied": Item("action", "A historical treatment entry was logged.", "Review the historical entry with a qualified adviser."),
    "irrigation": Item("action", "An irrigation event was logged.", "Check the timestamp on the recorded irrigation event."),
    "fallow": Item("action", "A fallow interval was logged at season end.", "Verify the recorded end-of-season status."),
    "scouting_visit": Item("action", "A follow-up scouting visit was recorded.", "Review or update the scouting log."),
    "no_intervention": Item("action", "No intervention was recorded for the visit.", "Review the recorded follow-up status."),
}


def validate_items(items: Sequence[str]) -> None:
    """Reject invalid event IDs with an optional close-match hint."""
    if not items:
        raise ValueError("Enter at least one item.")
    for item in items:
        if item not in ITEMS:
            close = get_close_matches(item, ITEMS, n=3, cutoff=0.55)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise ValueError(f"Unknown item: {item}.{hint}")


@dataclass(frozen=True)
class FieldHistory:
    """A field's observed visits, with event order preserved inside each visit."""

    field_id: str
    seasons: tuple[tuple[str, ...], ...]

    def validate(self) -> None:
        if not self.field_id or not self.seasons:
            raise ValueError("Each field needs an ID and at least one season.")
        for events in self.seasons:
            validate_items(events)
            if len(events) != len(set(events)):
                raise ValueError(f"{self.field_id} has a duplicate event in one season.")

    def baskets(self) -> tuple[frozenset[str], ...]:
        """Return independent unordered season baskets for pattern mining."""
        return tuple(frozenset(season) for season in self.seasons)

    def sequence(self) -> tuple[str, ...]:
        """Flatten season windows in chronological order for causal modeling."""
        return tuple(event for season in self.seasons for event in season)


CROPS = tuple(name for name, item in ITEMS.items() if item.kind == "crop")
ROTATION = {
    "wheat": ("chickpea", "soybean"), "rice": ("maize", "soybean"),
    "maize": ("soybean", "wheat"), "soybean": ("maize", "wheat"),
    "cotton": ("maize", "chickpea"), "chickpea": ("wheat", "cotton"),
}


def generate_fields(count: int = 1200, seed: int = 42) -> list[FieldHistory]:
    """Generate noisy field logs for a fictional farm, without treatment guidance."""
    if count < 1:
        raise ValueError("Field count must be positive.")
    rng = random.Random(seed)
    fields = []
    for number in range(1, count + 1):
        crop = rng.choice(CROPS)
        regime = rng.choice(("humid_season", "dry_season", "mild_season"))
        seasons = []
        for _ in range(rng.randint(4, 6)):
            weather = regime if rng.random() < 0.7 else rng.choice(
                ("humid_season", "dry_season", "mild_season")
            )
            humid, dry = weather == "humid_season", weather == "dry_season"
            events = [crop]
            if rng.random() < (0.60 if humid and crop in {"wheat", "soybean", "cotton"} else 0.12):
                events.append("aphids_observed")
            if rng.random() < (0.72 if humid and crop == "wheat" else 0.04):
                events.append("leaf_rust")
            if rng.random() < (0.48 if humid and crop in {"maize", "rice"} else 0.025):
                events.append("stem_borer")
            if rng.random() < (0.38 if crop == "cotton" else 0.015):
                events.append("boll_damage")
            if rng.random() < (0.67 if dry else 0.06):
                events.append("drought_stress")
            events.append(weather)
            if "leaf_rust" in events and rng.random() < 0.77:
                events.append("fungicide_applied")
            elif "drought_stress" in events and rng.random() < 0.76:
                events.append("irrigation")
            elif len(events) > 2 and rng.random() < 0.80:
                events.append("scouting_visit")
            else:
                events.append("no_intervention")
            if rng.random() < (0.22 if dry else 0.08):
                events.append("fallow")
            seasons.append(tuple(events))
            crop = rng.choices(ROTATION[crop], weights=(4, 1))[0] if rng.random() < 0.87 else rng.choice(CROPS)
        fields.append(FieldHistory(f"field-{number:04d}", tuple(seasons)))
    return fields


def split_fields(fields: Sequence[FieldHistory], seed: int = 42) -> dict[str, list[FieldHistory]]:
    """Allocate whole fields to deterministic train, validation, and test splits."""
    if len(fields) < 3 or len({field.field_id for field in fields}) != len(fields):
        raise ValueError("Need at least three fields with distinct IDs.")
    rows = sorted(fields, key=lambda field: field.field_id)
    random.Random(seed).shuffle(rows)
    held_out = max(1, int(len(rows) * 0.15))
    return {"train": rows[2 * held_out:], "validation": rows[:held_out], "test": rows[held_out:2 * held_out]}


# --- Milestone 2: Mine frequent co-occurring patterns using `thilimarketbasket`.
# Hint: Use `thilimarketbasket` to find frequent co-occurring patterns.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Train a small LM on these sequences for next-item prediction.
# Hint: Train a small LM on the generated sequences for next-item prediction.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate model's predictions on new, unseen agricultural sequences.
# Hint: Evaluate the model's predictions on a held-out set of agricultural sequences.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement a CLI to query likely next items based on current observations.
# Hint: Implement a command-line interface to query the trained model.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
