import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )
    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = dict()
    num_of_pages = len(corpus)

    # No outgoing links
    if len(corpus[page]) == 0:
        for p in corpus:
            distribution[p] = 1 / num_of_pages
    else:
        chance_for_all = (1 - damping_factor) / num_of_pages
        chance_for_linked = damping_factor / len(corpus[page])

        for site in corpus:
            if site in corpus[page]:
                distribution[site] = chance_for_linked + chance_for_all
            else:
                distribution[site] = chance_for_all

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    distribution = {key: 0 for key in corpus}
    sample = random.choice(list(corpus.keys()))

    for _ in range(n):
        distribution[sample] += 1

        sample_dis = transition_model(corpus, sample, damping_factor)

        # Representing the distribution on a scale of 0 to 1
        # Choose a random number between 0 and 1,
        # count up the pages' chances,
        # if the sum falls into to the range of the current sample, return it
        sample_num = (np.random.rand(1))
        c = 0
        for key in sample_dis:
            c += sample_dis[key]
            if c >= sample_num:
                sample = key
                break

    for key in distribution:
        distribution[key] /= n

    return distribution


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    def PR(page, distribution):

        random_vis = (1 - damping_factor) / len(local_corpus)
        linked_vis = 0

        for link in local_corpus:
            if page in local_corpus[link]:
                linked_vis += distribution[link] / len(local_corpus[link])

        return random_vis + (linked_vis * damping_factor)

    # Initializing the starer distributions
    distribution = {key: 1 / len(corpus) for key in corpus}
    old_distribution = {key: np.inf for key in corpus}

    # Checking for pages with no links,
    # and making a local copy of corpus to avoid modifying the original
    local_corpus = corpus.copy()
    for page in corpus:
        if len(corpus[page]) == 0:
            local_corpus[page] = set(key for key in corpus.keys())

    while True:
        for page in local_corpus:
            distribution[page] = PR(page, distribution)

        # Utilizing numpy's high precision subtract method for threshold
        changes = np.subtract(
            [old_distribution[page] for page in old_distribution], [distribution[page] for page in distribution]
        )

        if max(abs(diff) for diff in changes) < 0.001:
            break

        old_distribution = distribution.copy()

    return distribution


if __name__ == "__main__":
    main()
