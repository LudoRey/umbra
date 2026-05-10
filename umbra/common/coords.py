from dataclasses import dataclass

@dataclass
class Region:
    width: int
    height: int
    left: int = 0
    top: int = 0

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> 'Region':
        return cls(width=shape[1], height=shape[0])
    
    @property
    def right(self) -> int:
        return self.left + self.width
    
    @property
    def bottom(self) -> int:
        return self.top + self.height