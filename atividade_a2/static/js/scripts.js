document.addEventListener('DOMContentLoaded', () => {
    const valueChipsContainer = document.getElementById('valores-disponiveis');
    const attributeInputs = document.querySelectorAll('.attribute-input');
    const submitBtn = document.getElementById('submit-btn');

    const updateSubmitButtonState = () => {
        const allFieldsFilled = Array.from(attributeInputs).every(input => input.value !== '');
        submitBtn.disabled = !allFieldsFilled;
    };

    const findFirstEmptyInput = () => {
        return Array.from(attributeInputs).find(input => input.value === '');
    };

    const createChip = (value) => {
        const newChip = document.createElement('span');
        newChip.classList.add('value-chip');
        newChip.dataset.value = value;
        newChip.textContent = value;
        
        newChip.addEventListener('click', () => {
            const firstEmpty = findFirstEmptyInput();
            if (firstEmpty) {
                firstEmpty.value = newChip.dataset.value;
                firstEmpty.classList.add('filled');
                newChip.remove();
                updateSubmitButtonState();
            }
        });
        return newChip;
    };

    const sortChips = () => {
        const chips = Array.from(valueChipsContainer.children);
        chips.sort((a, b) => parseInt(a.dataset.value) - parseInt(b.dataset.value));
        chips.forEach(chip => valueChipsContainer.appendChild(chip));
    };

    document.querySelectorAll('.value-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const firstEmpty = findFirstEmptyInput();
            if (firstEmpty) {
                firstEmpty.value = chip.dataset.value;
                firstEmpty.classList.add('filled');
                chip.remove();
                updateSubmitButtonState();
            }
        });
    });


    attributeInputs.forEach(input => {
        input.addEventListener('click', () => {
            if (input.classList.contains('filled')) {
                const returnedValue = input.value;
                input.value = '';
                input.classList.remove('filled');
                
                const newChip = createChip(returnedValue);
                valueChipsContainer.appendChild(newChip);
                sortChips();
                updateSubmitButtonState();
            }
        });
    });

    updateSubmitButtonState();
});