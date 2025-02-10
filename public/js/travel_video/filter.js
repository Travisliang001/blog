<script>
document.getElementById('searchInput').addEventListener('input', function() {
    const searchQuery = this.value.toLowerCase();
    filterVideos(searchQuery);
});

    </script>
