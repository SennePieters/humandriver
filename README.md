A Python library for human-like mouse and keyboard automation, designed to work with `zendriver`.

## Features

- **Human-like Mouse Movement**: Uses WindMouse algorithms, and physiological jitter to simulate realistic cursor movement.
- **Human-like Typing**: Simulates typing speeds, typos, corrections, and thinking pauses.
- **Visual Debugging**: Generate heatmaps or trajectory JPEGs of mouse movements.

## Installation

```bash
pip install humandriver @ git+https://github.com/SennePieters/humandriver.git
```

## Usage

```python
from humandriver import move_to_element, type_in_element

# Example usage with zendriver page object
await move_to_element(page, selector, click=True)
await type_in_element(page, "Hello World")
```
