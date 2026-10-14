"""Independent checks of boundary, gradient, generation and reload contracts."""

from pathlib import Path
import tempfile
import unittest

import torch

from storyseed.bigram import BigramLanguageModel
from storyseed.checkpoint import load_checkpoint, save_checkpoint
from storyseed.data import make_windows, transition_pairs
from storyseed.generation import generate
from storyseed.loss import next_token_loss
from storyseed.tokenizer import CharacterTokenizer


class PhaseOneContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.tokenizer = CharacterTokenizer.fit(["ab", "ba"])
        self.model = BigramLanguageModel(3)

    def test_independent_documents_do_not_create_end_to_start_pairs(self):
        x, y = transition_pairs(["ab", "ba"], self.tokenizer)
        self.assertEqual(x.tolist(), [[0, 1, 1, 0]])
        self.assertEqual(y.tolist(), [[1, 2, 0, 2]])

    def test_last_valid_window_has_an_observed_end_target(self):
        x, y, locations = make_windows(["abba"], self.tokenizer, 2)
        self.assertEqual(locations, [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(x[-1].tolist(), [1, 0])
        self.assertEqual(y[-1].tolist(), [0, 2])
        with self.assertRaises(ValueError):
            make_windows(["a"], self.tokenizer, 3)

    def test_hand_gradient_changes_only_the_selected_row(self):
        with torch.no_grad():
            self.model.scores.zero_()
        loss = next_token_loss(self.model(torch.tensor([[1]])), torch.tensor([[0]]))
        loss.backward()
        expected = torch.zeros(3, 3)
        expected[1] = torch.tensor([-2/3, 1/3, 1/3])
        self.assertTrue(torch.allclose(self.model.scores.grad, expected))

    def test_generation_stops_at_eos_without_displaying_the_marker(self):
        with torch.no_grad():
            self.model.scores.fill_(-100)
            self.model.scores[0, 2] = 100
        result = generate(self.model, self.tokenizer, "a", top_k=1)
        self.assertEqual(result["text"], "a")
        self.assertEqual(result["generated_token_ids"], [2])
        self.assertEqual(result["stop_reason"], "eos")

    def test_sampling_repeats_and_preserves_model_mode(self):
        self.model.train()
        first = generate(self.model, self.tokenizer, "a", seed=8)
        second = generate(self.model, self.tokenizer, "a", seed=8)
        self.assertEqual(first, second)
        self.assertTrue(self.model.training)
        zero = generate(self.model, self.tokenizer, "a", max_new_tokens=0)
        self.assertEqual(zero["text"], "a")

    def test_invalid_prompt_and_decoding_settings_fail_clearly(self):
        for settings in ({"prompt": ""}, {"prompt": "!"}, {"prompt": "a", "temperature": 0},
                         {"prompt": "a", "temperature": float("nan")}, {"prompt": "a", "top_k": 4},
                         {"prompt": "a", "max_new_tokens": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                generate(self.model, self.tokenizer, **settings)

    def test_checkpoint_restores_weights_tokens_and_seeded_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model, self.tokenizer, {"note": "test fixture, not training"})
            loaded, tokenizer, _ = load_checkpoint(path)
            self.assertTrue(torch.equal(self.model.scores, loaded.scores))
            self.assertEqual(self.tokenizer.tokens, tokenizer.tokens)
            self.assertEqual(generate(self.model, self.tokenizer, "ab", seed=5),
                             generate(loaded, tokenizer, "ab", seed=5))

    def test_corrupt_shape_and_nonfinite_checkpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for shape in (True, False):
                save_checkpoint(path, self.model, self.tokenizer, {})
                payload = torch.load(path, weights_only=True)
                if shape:
                    payload["state_dict"]["scores"] = torch.zeros(2, 2)
                else:
                    payload["state_dict"]["scores"][0, 0] = float("nan")
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
