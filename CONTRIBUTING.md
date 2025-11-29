# Contributing to Vocal Tract Segmentation

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: How to recreate the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, OS, GPU/CPU
- **Error Messages**: Full error traceback if applicable

### Suggesting Enhancements

For feature requests or improvements:

- Check existing issues to avoid duplicates
- Clearly describe the enhancement
- Explain the use case and benefits
- Consider implementation complexity

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, documented code
   - Follow existing code style
   - Add docstrings to functions
   - Update documentation if needed

4. **Test your changes**
   - Ensure code runs without errors
   - Test with sample data
   - Verify backward compatibility

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Describe what changed and why
   - Reference related issues
   - Include test results if applicable

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/vocal_tract_segmentation.git
cd vocal_tract_segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Comment complex logic

Example:
```python
def segment_vocal_tract(image, model):
    """
    Segment vocal tract structures in MRI image.
    
    Args:
        image: MRI image array of shape (256, 256, 1)
        model: Trained U-Net model
    
    Returns:
        Segmentation mask of shape (256, 256, 7)
    """
    # Implementation
    pass
```

## Testing

Before submitting:

1. Test with sample data
2. Verify no breaking changes
3. Check for memory leaks (for large datasets)
4. Test on different Python versions if possible

## Documentation

Update documentation when:

- Adding new features
- Changing function signatures
- Modifying configuration options
- Adding new dependencies

## Areas for Contribution

We welcome contributions in:

### High Priority
- [ ] Unit tests for core modules
- [ ] 3D volumetric segmentation support
- [ ] Additional data augmentation techniques
- [ ] Pre-trained model weights
- [ ] Performance optimization

### Medium Priority
- [ ] Additional evaluation metrics (Hausdorff distance, IoU)
- [ ] Model ensemble capabilities
- [ ] Export to ONNX format
- [ ] Real-time inference optimization
- [ ] Interactive visualization tools

### Documentation
- [ ] Tutorial notebooks
- [ ] Video tutorials
- [ ] API documentation improvements
- [ ] Use case examples
- [ ] Clinical application guides

### Research
- [ ] Alternative architectures (ResNet, DenseNet)
- [ ] Transfer learning experiments
- [ ] Multi-modal input support
- [ ] Uncertainty quantification
- [ ] Active learning strategies

## Questions?

- Open an issue for questions
- Check existing issues and documentation
- Contact maintainers via email

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the code, not the person
- Accept constructive criticism gracefully

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to vocal tract segmentation research! 🎉
