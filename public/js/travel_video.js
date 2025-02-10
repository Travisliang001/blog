<script>
    function scrollDown() {
        // Hide background image and arrow-circle
        var backgroundImage = document.querySelector('.background-image');
        var arrowCircle = document.querySelector('.arrow-circle');

        if (backgroundImage && arrowCircle) {
            backgroundImage.style.opacity = '0';
            arrowCircle.style.display = 'none';
        }

        // Scroll down to the contents
        var contents = document.querySelector('.contents');
        if (contents) {
            contents.scrollIntoView({ behavior: 'smooth' });
        }
    }
</script>
