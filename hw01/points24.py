from copy import deepcopy

#
class processRecord:
    def __init__(self, cards):
        self.is_last_combine = False
        self.solution = {}
        # initialize here, using total cards length
        for i in range(1 + len(cards)):
            self.solution[i] = {}

    def add_combination(self, cards, i, j):
        self.solution[len(cards)]['cards'] = cards
        self.solution[len(cards)]['combined_index'] = [i, j]
        self.solution[len(cards)]['combined_cards'] = [cards[i], cards[j]]

    def add_operation(self, i, two_cards, cards):
        # Operation in order, operated cards indicated by solution[len(cards)]['combined_index']
        self.solution[len(cards)]['operation'] = i
        # just for test
        self.solution[len(cards)]['operated_cards'] = two_cards

    def print_sol(self):
        print(self.solution)


def one_operation(card1, card2, operation_index):
    if operation_index == 0:
        return card1 + card2
    elif operation_index == 1:
        return card1 * card2
    elif operation_index == 2:
        return card1 - card2
    elif operation_index == 3:
        if card2 != 0:
            return card1 / card2
        else:
            return None
    elif operation_index == 4:
        return card2 - card1
    elif operation_index == 5:
        if card1 != 0:
            return card2 / card1
        else:
            return None


def operation(cards, two_cards, goal, process):
    if process.is_last_combine:
        # last round, all cards combined, if the result is goal then print
        for i in range(6):
            this_process = deepcopy(process)
            result = one_operation(two_cards[0], two_cards[1], i)
            # catch division by zero
            if result is None:
                continue
            if abs(result - goal) < 1e-3:
                this_process.add_operation(i, two_cards, cards)
                this_process.print_sol()
    else:
        for i in range(6):
            # new branches: given 2 cards and branch in different operations
            # process and cards need to be copied so that next iteration will not be affected
            new_card = one_operation(two_cards[0], two_cards[1], i)
            # catch division by zero
            if new_card is None:
                continue
            this_cards = cards.copy()
            this_cards.remove(two_cards[0])
            this_cards.remove(two_cards[1])
            this_cards.append(new_card)
            # record in process!
            this_process = deepcopy(process)
            this_process.add_operation(i, two_cards, cards)
            combine(this_cards, goal, this_process)
    return

def combine(cards, goal, process):
    if len(cards) == 2:
        two_cards = cards.copy()
        # no cards left, 2 cards need to combine
        cards = []
        # last turn: call operation to check if meets goal
        process.is_last_combine = True
        operation(cards, two_cards, goal, process)
    else:
        # new card is appended at the end
        # combination: approximately C_{len(cards)}^2/2
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                card1 = cards[i]
                # print(len(cards)-j)
                card2 = cards[j]
                # new branches: given cards, branch in different operations
                # process needs to be copied (cards are not edited here) so that next iteration will not be affected
                this_process = deepcopy(process)
                # record in process!
                this_process.add_combination(cards, i, j)
                operation(cards, [card1, card2], goal, this_process)
    return

def main():
    # input 4 cards
    cards = []
    for i in range(4):
        cards.append(float(input(f'Type in the point of the {i+1}-th card')))
    assert len(cards) == 4, 'Invalid size of input'

    goal = 24.
    process = processRecord(cards)
    combine(cards, goal, process)

def test():
    process = processRecord([1,2,3,4])
    process.add_combination([1,2,3,4], 1,2)
    process.add_operation(1, [2, 3], [1,2,3,4])
    print(process.solution)
    return 0

if __name__ == '__main__':
    main()
