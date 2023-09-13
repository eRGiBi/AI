import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return set(self.cells)
        else:
            return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        else:
            return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        for sentence_cell in self.cells:
            if sentence_cell == cell:
                self.cells.remove(sentence_cell)
                self.count -= 1
                break

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        for sentence_cell in self.cells:
            if sentence_cell == cell:
                self.cells.remove(sentence_cell)
                break


def is_it_subset(sentence, sub_sentence):
    """
    Checks if a sentence is a subset of another.
    """
    return set(sub_sentence.cells).issubset(sentence.cells)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        # Mark the cell as a move that has been made
        self.moves_made.add(cell)

        # Mark the cell as safe
        self.mark_safe(cell)

        # Adding the new sentence without counting known neighboring mines

        self.knowledge.append(Sentence(self.get_neighbors(cell), count - self.count_of_neighboring_mines(cell)))

        # Iterating through the knowledge base for finding subsets
        for i in self.knowledge:
            for j in self.knowledge:
                if (i != j
                        and len(i.cells) != 0
                        and len(j.cells) != 0
                        and is_it_subset(i, j)):

                    n = Sentence(i.cells - j.cells, i.count - j.count)

                    if n not in self.knowledge:
                        self.knowledge.append(n)

        # Marking known safes and mines
        for sentence in self.knowledge:
            for safe in set(sentence.known_safes()):
                if safe not in self.safes:
                    self.mark_safe(safe)

        for sentence in self.knowledge:
            for mine in set(sentence.known_mines()):
                if mine not in self.mines:
                    self.mark_mine(mine)

        # Removing sentences which have been used up
        for sentence in self.knowledge:
            if len(sentence.cells) == 0 or sentence.cells == set():
                self.knowledge.remove(sentence)

    def get_neighbors(self, cell):
        """
        Returns the list of unknown  neighboring  cells.
        """
        neighbors = []
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue
                else:
                    neighbor = (i, j)
                    if (0 <= i < self.width) and (0 <= j < self.height):
                        if (neighbor not in self.mines and neighbor not in self.safes
                                and neighbor not in self.moves_made):
                            neighbors.append(neighbor)

        return neighbors

    def count_of_neighboring_mines(self, cell):
        """
        Returns the count of known neighboring mines.
        """
        count = 0
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                if (i, j) == cell:
                    continue
                else:
                    neighbor = (i, j)
                    if neighbor in self.mines:
                        count += 1
        return count

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for safe in self.safes:
            if safe not in self.moves_made:
                return safe
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        if (self.height * self.width) - len(self.mines) == len(self.moves_made):
            return None

        move = (random.randrange(self.width - 1), random.randrange(self.height - 1))
        while move in self.moves_made or move in self.mines:
            move = (random.randrange(self.width - 1), random.randrange(self.height - 1))

        return move
