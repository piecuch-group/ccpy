# The ideal input file should consist of three items

run = Runner(threads=4,
             max_memory="5g")

# For ground state
mol = Molecule()

inp = EnergyInput(
    molecule=mol,
    method=CCMethod(truncation=2)
)

result = run(inp)

# For excited states
inp = EnergyInput(
    molecule=mol,
    method=EOMCCMethod()
)
