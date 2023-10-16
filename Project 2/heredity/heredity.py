import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])
    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):

    

    

    # def probability_on_gene_type(x, l):
    #     p = 1
    #     for person in l:
    #         p *= PROBS["gene"][x]
    #         if person in have_trait:
    #             p *= PROBS["trait"][x][True]
    #         else:
    #             p *= PROBS["trait"][x][False]
    #     return p
    #
    # zero_gene = []
    # for person in people:
    #     if person not in one_gene and person not in two_genes:
    #         zero_gene.append(person)
    # # c0 = probability_on_gene_type(0, zero_gene)
    # ch = 1
    # for person in people:
    #     if person in one_gene:
    #         num = 1
    #     elif person in two_genes:
    #         num = 2
    #     else:
    #         num = 0
    #
    #     mother = people[person]["mother"]
    #     father = people[person]["father"]
    #     mother_chance = 1
    #     father_chance = 1
    #
    #     if mother is None:
    #         mother_chance *= PROBS["gene"][num]
    #         father_chance *= 1 - PROBS["gene"][num]
    #     elif mother in zero_gene:
    #         father_chance *= 1 - PROBS["mutation"]
    #         mother_chance *= PROBS["mutation"]
    #     elif mother in two_genes:
    #         father_chance *= PROBS["mutation"]
    #         mother_chance *= 1 - PROBS["mutation"]
    #
    #     if father is None:
    #         father_chance *= PROBS["gene"][num]
    #         mother_chance *= 1 - PROBS["gene"][num]
    #     elif father in zero_gene:
    #         father_chance *= PROBS["mutation"]
    #         mother_chance *= 1 - PROBS["mutation"]
    #     elif father in two_genes:
    #         father_chance *= 1 - PROBS["mutation"]
    #         mother_chance *= PROBS["mutation"]
    #
    #     if father in one_gene or mother in one_gene:
    #         father_chance *= 0.5
    #         mother_chance *= 0.5
    #
    #     ch *= PROBS["trait"][num][person in have_trait] * (father_chance + mother_chance)
    #
    # print(ch )
    # return ch


# def joint_probability2(people, one_gene, two_genes, have_trait):
#     """
#     Compute and return a joint probability.
#
#     The probability returned should be the probability that
#         * everyone in set `one_gene` has one copy of the gene, and
#         * everyone in set `two_genes` has two copies of the gene, and
#         * everyone not in `one_gene` or `two_gene` does not have the gene, and
#         * everyone in set `have_trait` has the trait, and
#         * everyone not in set` have_trait` does not have the trait.
#     """
#
#     def probability_on_gene_type(x, lst):
#         chance = 1
#         for person in lst:
#             if people[person]["mother"] is None and people[person]["father"] is None:
#                 chance *= PROBS["gene"][x]
#                 chance *= PROBS["trait"][x][person in have_trait]
#             # TODO different parents?
#             else:
#                 chance *= PROBS["gene"][x]
#                 chance *= PROBS["trait"][x][person in have_trait]
#                 chance *= get_parent_probability(person, x)
#         return chance
#
#     def get_parent_probability2(person, x):
#
#         mother = people[person]["mother"]
#         father = people[person]["father"]
#         mother_chance = 1
#         father_chance = 1
#
#         if mother is None:
#             mother_chance *= PROBS["gene"][x]
#             father_chance *= 1 - PROBS["gene"][x]
#         elif mother in zero_gene:
#             father_chance *= 1 - PROBS["mutation"]
#             mother_chance *= PROBS["mutation"]
#         elif mother in two_genes:
#             father_chance *= PROBS["mutation"]
#             mother_chance *= 1 - PROBS["mutation"]
#
#         if father is None:
#             father_chance *= PROBS["gene"][x]
#             mother_chance *= 1 - PROBS["gene"][x]
#         elif father in zero_gene:
#             father_chance *= PROBS["mutation"]
#             mother_chance *= 1 - PROBS["mutation"]
#         elif father in two_genes:
#             father_chance *= 1 - PROBS["mutation"]
#             mother_chance *= PROBS["mutation"]
#
#         if father in one_gene or mother in one_gene:
#             father_chance *= 0.5
#             mother_chance *= 0.5
#
#         return father_chance + mother_chance
#
#     # print(people)
#     print(one_gene)
#     print(two_genes)
#     print(have_trait)
#
#     zero_gene = []
#     for person in people:
#         if person not in one_gene and person not in two_genes:
#             zero_gene.append(person)
#
#     # # Zero gene --------------------------------------------------
#     # zero_gene = []
#     # for person in people:
#     #     if person not in one_gene and person not in two_genes:
#     #         zero_gene.append(person)
#     # # print(zero_gene)
#     #
#     # p_0 = probability_on_gene_type(0, zero_gene)
#     # # print(p_0)
#     #
#     # # One gene ---------------------------------------------
#     # p_1 = 1
#     # for person in one_gene:
#     #     p_1 *= PROBS["trait"][1][person in have_trait] * (get_parent_probability(person, 1))
#     #
#     # # Two gene -----------------------------------------------------
#     # p_2 = probability_on_gene_type(2, two_genes)
#     # print(p_0, p_2, p_1)
#     # print(p_1 * p_2 * p_0)
#     # return p_1 * p_2 * p_0
#
#     final_prob = 1
#     p_1 = 1
#     p_2 = 1
#     p_0 = 1
#     for person in people:
#         p_1 = 1
#         p_2 = 1
#         p_0 = 1
#
#         if person in one_gene:
#             p_1 = PROBS["trait"][1][person in have_trait] * get_parent_probability(person, 1)
#             p_1 *= PROBS["gene"][1]
#         elif person in two_genes:
#             p_2 = PROBS["trait"][2][person in have_trait] * get_parent_probability(person, 2)
#             p_2 *= PROBS["gene"][2]
#         else:
#             p_0 = PROBS["trait"][0][person in have_trait] * get_parent_probability(person, 0)
#             p_0 *= PROBS["gene"][0]
#         final_prob *= p_1 * p_2 * p_0
#
#     print(final_prob)
#     return final_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:

        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:

        s = sum(probabilities[person]["gene"].values())
        multi = 1 / s
        for dis in probabilities[person]["gene"]:
            probabilities[person]["gene"][dis] *= multi

        s = sum(probabilities[person]["trait"].values())
        multi = 1 / s
        for dis in probabilities[person]["trait"]:
            probabilities[person]["trait"][dis] *= multi


if __name__ == "__main__":
    p = joint_probability({
        'Harry': {'name': 'Harry', 'mother': 'Lily', 'father': 'James', 'trait': None},
        'James': {'name': 'James', 'mother': None, 'father': None, 'trait': True},
        'Lily': {'name': 'Lily', 'mother': None, 'father': None, 'trait': False}
    }, {"Harry"}, {"James"}, {"James"})

    main()
