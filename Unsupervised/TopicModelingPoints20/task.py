import os
import gdown
import numpy as np
import pandas as pd

from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
from tqdm import tqdm


class PLSA:
    def __init__(self, counts, num_topics, seed):
        """
        Initialize the PLSA model.

        :param counts: A 2D numpy array with shape (num_words, num_documents) representing word counts.
        :param num_topics: The number of topics (T).
        :param seed: Random seed for initialization.
        """
        self.counts = counts
        self.num_topics = num_topics
        self.num_docs = counts.shape[1]
        self.num_words = counts.shape[0]
        self.total_words = np.sum(counts)
        self._initialize_params(seed)

    def _initialize_params(self, seed):
        np.random.seed(seed)

        self.Phi = np.random.random((self.num_words, self.num_topics))
        self.Phi /= np.sum(self.Phi, axis=0, keepdims=True)

        self.Theta = np.random.random((self.num_topics, self.num_docs))
        self.Theta /= np.sum(self.Theta, axis=0, keepdims=True)

    def _update_parameters(self):
        Theta_new = np.zeros_like(self.Theta)
        Phi_new = np.zeros_like(self.Phi)
        topic_probability_matrix = np.dot(self.Phi, self.Theta)
        word_counts_per_doc = np.sum(self.counts, axis=0)

        for t in range(self.num_topics):
            prob_topic = np.divide(
                self.Phi[:, [t]] * self.Theta[[t], :],
                topic_probability_matrix,
                where=(topic_probability_matrix > 0)
            )
            topic_distribution = np.einsum('wd, wd -> d', self.counts, prob_topic)
            topic_distribution = np.divide(topic_distribution, word_counts_per_doc, where=(word_counts_per_doc > 0))
            Theta_new[[t], :] = np.where(word_counts_per_doc == 0, 0, topic_distribution)

            word_distribution = np.einsum('wd, wd -> w', self.counts, prob_topic)
            n_t = np.sum(word_distribution, axis=0, keepdims=True)
            phi = np.divide(word_distribution, n_t, where=(n_t > 0))
            Phi_new[:, t] = np.where(n_t == 0, 0, phi)

        self.Theta = Theta_new
        self.Phi = Phi_new

    def perplexity(self):
        topic_probability_matrix = np.dot(self.Phi, self.Theta)
        log_likelihood = np.multiply(self.counts,
                                     np.log(topic_probability_matrix,
                                            where=(topic_probability_matrix > 0))
                                     )
        return np.exp(-np.sum(log_likelihood) / self.total_words)


def download_if_not_exists(file_id, output_file):
    """
    Download file from Google Drive if it doesn't exist locally

    Args:
        file_id (str): Google Drive file ID
        output_file (str): Local filename to save to
    """
    if not os.path.exists(output_file):
        print(f"Downloading {output_file}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_file, quiet=False)
        print("Download complete!")
    else:
        print(f"File {output_file} already exists, skipping download")


def data_download():
    file_id = "1GD6qjaYqKJV5g_i5wpaCpa_e_wzZjPQg"
    download_if_not_exists(file_id, "vacancy_skills.csv")
    file_id = "18sUrPye4er8uIDbsEubnS9lLFmJ-0kBQ"
    download_if_not_exists(file_id, "vacancies_train.csv")
    file_id = "1XPhtfDEiVSKbuKIOv_q2wIsO8YD1Upo6"
    download_if_not_exists(file_id, "vacancies_test.csv")

def main():
    data_download()
    vacancies = pd.read_csv('vacancies_train.csv', index_col=0)
    vacancies['name_descr'] = vacancies.apply(lambda r: r['name'] + ' ' + r['description'], axis=1)

    tokenizer = WordPunctTokenizer()
    lines = [' '.join(tokenizer.tokenize(
        BeautifulSoup(line, features="html.parser").text.lower()))
             for line in tqdm(vacancies['name_descr'].sample(1000))]

    vectorizer = CountVectorizer(max_df=0.95, min_df=100)
    X = vectorizer.fit_transform(lines)
    counts = X.todense().T
    np.save("matrix", counts)


if __name__ == '__main__':
    main()
