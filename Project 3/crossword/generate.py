import sys

import numpy as np

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable in self.domains:
            for word in self.domains[variable].copy():
                if len(word) != variable.length:
                    self.domains[variable].remove(word)
    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revision_was_made = False
        overlap = self.crossword.overlaps[x, y]

        if overlap is None:
            return False

        for x_word in self.domains[x].copy():
            if all(x_word[overlap[0]] != y_word[overlap[1]] for y_word in self.domains[y]):
                self.domains[x].remove(x_word)
                revision_was_made = True

        return revision_was_made

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            queue = []
            for f in self.crossword.variables:
                for s in self.crossword.variables:
                    if s != f:
                        queue.append((s, f))
        else:
            queue = arcs
            print(arcs)

        while len(queue) != 0:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False
                else:
                    for z in self.crossword.neighbors(x):
                        if z == y:
                            continue
                        queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for key in self.crossword.variables:
            if key not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        values = set()
        for variable in assignment:

            # If the variable's length not fit
            if len(assignment[variable]) != variable.length:
                return False
            # If there is a duplicate in the assignment regardless of position
            if assignment[variable] in values:
                return False

            for neighbor in self.crossword.neighbors(variable):
                if assignment.get(neighbor):
                    # If neighbors have the same value
                    if assignment[variable] == assignment[neighbor]:
                        return False
                    # If the overlap doesn't match
                    overlap = self.crossword.overlaps[variable, neighbor]
                    if assignment[variable][overlap[0]] != assignment[neighbor][overlap[1]]:
                        return False

            values.add(assignment[variable])

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        values = {}
        neighbors = self.crossword.neighbors(var)

        for value in self.domains[var]:
            count = 0

            # new_assignment = assignment.copy()
            # new_assignment[var] = value

            # if self.consistent(new_assignment):
            #
            #     for neighbor in neighbors:
            #         if neighbor in assignment:
            #             continue
            #         else:
            #             # print("domain of neg", self.domains[neighbor])
            #             for possible_word in self.domains[neighbor]:
            #                 newer_assignment = new_assignment
            #                 newer_assignment[neighbor] = possible_word
            #                 if not self.consistent(newer_assignment):
            #                 # if possible_word == value:
            #                     count += 1

            # go tru neighbor values if overlap not gud  +1
            for neighbor in neighbors:
                if neighbor in assignment:
                    continue
                overlap = self.crossword.overlaps[var, neighbor]
                if overlap is not None:
                    for neighbor_value in self.domains[neighbor]:
                        if value[overlap[0]] != neighbor_value[overlap[1]]:
                            count += 1

            values[value] = count

        values = sorted(values.items(), key=lambda x: x[1])
        return {value[0] for value in values}

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        var = None
        value_count = np.Inf
        for variable in self.crossword.variables:
            if variable not in assignment:
                # return the variable with the fewest number of remaining values
                if len(self.domains[variable]) < value_count:
                    var = variable
                    value_count = len(self.domains[variable])
                # choose which variable has the largest degree (has the most neighbors)
                elif len(self.domains[variable]) == value_count:
                    if len(self.crossword.neighbors(variable)) > len(self.crossword.neighbors(var)):
                        var = variable
                        value_count = len(self.domains[variable])
        return var

    def backtrack2(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value

            if self.consistent(new_assignment):

                old_domains = self.domains
                self.enforce_node_consistency()

                result = self.backtrack(new_assignment)
                if result is not None:
                    return result

                else:
                    print("removed")
                    self.domains = old_domains
                    self.domains[var].remove(value)

        return None

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        # print("selce: ", var)
        #
        # print(self.domains[var])
        # print(self.order_domain_values(var, assignment))

        # for value in self.domains[var]:
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            original_domains = []
            old_domains = self.domains

            if self.consistent(new_assignment):

                old_domains = self.domains
                self.enforce_node_consistency()

                infe = []
                for ne in self.crossword.neighbors(var):
                    original_domains.append(self.domains[ne])
                    infe.append((ne, var))
                self.ac3(infe)

                # if self.ac3(infe):
                #     for i in range(len(self.crossword.neighbors(var))):
                #         assignment[list(self.crossword.neighbors(var))[i]] = (
                #             self.domains)[list(self.crossword.neighbors(var))[i]]

                result = self.backtrack(new_assignment)
                if result is not None:
                    return result
            else:
                print("removed")
                self.domains[var].remove(value)

                for i in range(len(self.crossword.neighbors(var))):
                    self.domains[self.crossword.neighbors(var)[i]] = original_domains[i]
                self.domains = old_domains

        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
