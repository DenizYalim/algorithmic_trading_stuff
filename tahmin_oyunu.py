import random


def set_game():
    global number
    number = random.randint(1, 100)


def guess_number():
    round = 1
    while True:
        print(f"Round {round}")
        try:
            guess = int(input("1 ile 100 arasında bir sayı tahmin edin: "))
            if guess < number:
                print("Daha yüksek bir sayı tahmin edin.")
            elif guess > number:
                print("Daha düşük bir sayı tahmin edin.")
            else:
                print(f"Tebrikler! Doğru tahmin ettiniz. Sayı {number} idi.")
                break
        except ValueError:
            print("That's not a valid number.")
        round += 1


if __name__ == "__main__":
    set_game()
    guess_number()
