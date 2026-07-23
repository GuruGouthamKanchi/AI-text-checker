import os
import sys
import tempfile
import unittest
import pandas as pd

# Ensure the src directory is in python search path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from clean_text import clean_text, split_into_sentences
from data_loaders import load_daigt, _normalize_topic
from dedupe import remove_near_duplicates
from split import topic_aware_split

class TestPreprocessingPipeline(unittest.TestCase):
    
    def test_clean_text(self):
        # Strips HTML tags
        self.assertEqual(clean_text("<p>Hello <b>world</b></p>"), "Hello world")
        # Cleans control characters and normalizes carriage returns
        self.assertEqual(clean_text("Hello\x00 world\r\n"), "Hello world")
        # Replaces smart quotes
        self.assertEqual(clean_text("“Hello” from ‘Antigravity’"), '"Hello" from \'Antigravity\'')
        # Strips markdown headers and bold/italic indicators
        self.assertEqual(clean_text("## Header\n**bold** and _italic_"), "Header\nbold and italic")
        # Normalizes multiple horizontal spaces
        self.assertEqual(clean_text("Too    much   space"), "Too much space")
        # Cleans out inline code tags
        self.assertEqual(clean_text("Code: `python -m unittest`"), "Code: python -m unittest")

    def test_split_into_sentences(self):
        text = "Hello world. This is a sentence! Does it work? Yes, indeed."
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertEqual(sentences[0], "Hello world.")
        self.assertEqual(sentences[2], "Does it work?")

    def test_normalize_topic(self):
        self.assertEqual(_normalize_topic("Car-free Cities! (USA)"), "carfree_cities_usa")
        self.assertEqual(_normalize_topic("   "), "unknown_topic")

    def test_load_daigt_defensive(self):
        # Test that the loader is defensive and maps various inputs correctly
        data = {
            'essay': ["This is a student essay about topic A.", "This is a chatgpt essay about topic B."],
            'generated': [0, 1],
            'prompt_name': ["Topic A", "Topic B"],
            'source': ["train_essays", "chatgpt_prompt_B"]
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "mock_daigt.csv")
            df.to_csv(csv_path, index=False)
            
            loaded_df = load_daigt(csv_path)
            self.assertEqual(len(loaded_df), 2)
            self.assertListEqual(list(loaded_df.columns), ['text', 'label', 'source', 'topic_id'])
            self.assertEqual(loaded_df.iloc[0]['text'], "This is a student essay about topic A.")
            self.assertEqual(loaded_df.iloc[0]['label'], 0)
            self.assertEqual(loaded_df.iloc[0]['topic_id'], "topic_a")
            self.assertEqual(loaded_df.iloc[1]['topic_id'], "topic_b")

    def test_dedupe_exact_and_fuzzy(self):
        # Set up a dataset with exact duplicates and near duplicates (Jaccard >= 0.9)
        data = {
            'text': [
                "This is a unique document representing text number one.",
                "This is a unique document representing text number one.", # Exact duplicate
                "This is a unique document representing text number two.", # Unique
                "This is a unique document representing text number one!", # Near-duplicate (Jaccard > 0.9)
                "Completely unrelated text for deduplication verification." # Unique
            ],
            'label': [0, 0, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        
        # Threshold 0.9 should remove both exact and the near-duplicate "text number one!"
        deduped = remove_near_duplicates(df, text_col="text", threshold=0.9)
        self.assertEqual(len(deduped), 3)
        self.assertNotIn("This is a unique document representing text number one!", deduped['text'].values)

    def test_topic_aware_split(self):
        # 100 rows across 5 unique topics
        data = {
            'text': [f"Essay content {i}" for i in range(100)],
            'topic_id': [f"topic_{i % 5}" for i in range(100)]
        }
        df = pd.DataFrame(data)
        
        # 80 / 10 / 10 proportions
        train_df, val_df, test_df = topic_aware_split(
            df, topic_col="topic_id", train_frac=0.8, val_frac=0.1, test_frac=0.1, random_seed=42
        )
        
        # Check that splits are completely disjoint by topic_id (leak-free)
        train_topics = set(train_df['topic_id'].unique())
        val_topics = set(val_df['topic_id'].unique())
        test_topics = set(test_df['topic_id'].unique())
        
        self.assertTrue(train_topics.isdisjoint(val_topics), "Leakage between train and val!")
        self.assertTrue(train_topics.isdisjoint(test_topics), "Leakage between train and test!")
        self.assertTrue(val_topics.isdisjoint(test_topics), "Leakage between val and test!")
        
        # All rows accounted for
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))

if __name__ == "__main__":
    unittest.main()
