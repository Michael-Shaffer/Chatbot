document.addEventListener('DOMContentLoaded', () => {
    const sky = document.querySelector('.sky');
    const starContainers = document.querySelectorAll('.star-container');
    const proximityThreshold = 75;

    // Store star positions so we don't recalculate them constantly
    let starPositions = [];
    function cacheStarPositions() {
        starPositions = []; // Clear cache before recalculating
        starContainers.forEach(container => {
            const rect = container.getBoundingClientRect();
            starPositions.push({
                label: container.querySelector('.star-label'),
                x: rect.left + (rect.width / 2),
                y: rect.top + (rect.height / 2),
            });
        });
    }
    // Initial calculation on page load
    cacheStarPositions();
    // Recalculate if the window is resized
    window.addEventListener('resize', cacheStarPositions);
});
