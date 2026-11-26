import os
import random
import re
import sys

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

    pages = list(corpus)
    page_count = len(pages)

    # A page with no outgoing links behaves as if it linked to every page.
    links = corpus[page] if corpus[page] else set(pages)

    distribution = {
        destination: (1 - damping_factor) / page_count
        for destination in pages
    }
    link_probability = damping_factor / len(links)
    for destination in links:
        distribution[destination] += link_probability

    return distribution


def sample_pagerank(corpus, damping_factor, n):

    pages = list(corpus)
    counts = {page: 0 for page in pages}

    # The first sample is a random page; later samples follow the transition
    # distribution from the page sampled immediately before.
    page = random.choice(pages)
    counts[page] += 1

    for _ in range(n - 1):
        distribution = transition_model(corpus, page, damping_factor)
        choices = list(distribution)
        weights = [distribution[choice] for choice in choices]
        page = random.choices(choices, weights=weights, k=1)[0]
        counts[page] += 1

    return {page: counts[page] / n for page in pages}


def iterate_pagerank(corpus, damping_factor):

    pages = list(corpus)
    page_count = len(pages)
    ranks = {page: 1 / page_count for page in pages}

    while True:
        updated_ranks = {}

        for page in pages:
            rank = (1 - damping_factor) / page_count

            for source in pages:
                links = corpus[source]
                if not links:
                    rank += damping_factor * ranks[source] / page_count
                elif page in links:
                    rank += damping_factor * ranks[source] / len(links)

            updated_ranks[page] = rank

        if all(
            abs(updated_ranks[page] - ranks[page]) < 0.001
            for page in pages
        ):
            return updated_ranks

        ranks = updated_ranks


if __name__ == "__main__":
    main()
