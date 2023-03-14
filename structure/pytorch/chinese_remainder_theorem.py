def coprimes():
    "Generating all coprime pairs"
    yield (2, 1)
    yield (3, 1)
    for m, n in coprimes():
        yield (2*m -n, m)
        yield (2*m + n, m)
        yield (m + 2*n, n)

def extended_euclid(a, b):
    if b == 0:
        return (1, 0)
    (x, y) = extended_euclid(b, a % b)
    k = a // b
    return (y, x - k * y)


class ChineseRemainderTheorem():
    def __init__(self, coprime1: int = 122, coprime2: int = 121):
        self.coprime1, self.coprime2 = coprime1, coprime2
        self.x, self.y = extended_euclid(coprime1, coprime2)

    def to_x(self, r1, r2):
        # (x, y) = extended_euclid(self.coprime1, self.coprime2)
        m = self.coprime1 * self.coprime2
        n = r2 * self.x * self.coprime1 + r1 * self.y * self.coprime2
        return (n % m + m) % m

    def to_r(self, x):
        # assert 0 <= x < self.coprime1 * self.coprime2
        return x % self.coprime1, x % self.coprime2



if __name__ == "__main__":

    # pairs = coprimes()
    # for _ in range(100000000):
    #     prime1, prime2 = next(pairs)
    #     if prime1 - prime2 < 3:
    #         print(prime1, prime2)

    #  possible coprime pair: (121, 122) (255, 256) (18, 17)
    coprime1 = 18
    coprime2 = 17
    crt = ChineseRemainderTheorem(coprime1, coprime2)

    for x in range(coprime1 * coprime2):
        r1, r2 = crt.to_r(x)
        assert crt.to_x(r1, r2) == x
        print(x, r1, r2)