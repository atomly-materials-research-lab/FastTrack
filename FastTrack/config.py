#from chgnet.model import CHGNetCalculator
from gptff.model.mpredict import ASECalculator
#from mace.calculators import mace_mp
#from ase.calculators.mixing import SumCalculator
#from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
#from mattersim.forcefield import MatterSimCalculator
#from sevenn.calculator import SevenNetCalculator

def Calculator ():

    #calc = CHGNetCalculator()
    #calc = CHGNetCalculator.from_file("yourpath")
    #d3_calc = TorchDFTD3Calculator()  
    #calc = SumCalculator([chgnet_calc, d3_calc])
    
    
    MODEL_WEIGHT = "yourpath/gptff_v1.pth"
    DEVICE = "cuda"
    calc = ASECalculator(MODEL_WEIGHT, DEVICE)
    
    #calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cuda")

    #model_path = "yourpath"
    #calc = mace_mp(model= model_path, dispersion=False, default_dtype="float64", device='cuda')
    
    #calc = SumCalculator([mace_calc, d3_calc])
    
    #calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')

    return calc
