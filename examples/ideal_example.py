# The ideal input file should consist of three items

# For ground state
mol = Molecule()

inp = EnergyInput(
    molecule=mol,
    method=CCMethod(truncation=2)
)

result = driver.run(inp)

# For excited states
inp = EnergyInput(
    molecule=mol,
    method=EOMCCMethod()
)