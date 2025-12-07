from dataclasses import dataclass

@dataclass
class Region:
    width: int
    height: int
    left: int = 0
    top: int = 0
    
@dataclass
class Point:
    x: float
    y: float