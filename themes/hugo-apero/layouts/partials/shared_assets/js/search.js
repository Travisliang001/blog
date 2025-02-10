document.addEventListener("DOMContentLoaded", function () {
  const searchButton = document.querySelector(".search-button");
  const searchBox = document.querySelector(".search-box");
  const searchInput = document.getElementById("search-input");
  const searchResults = document.getElementById("search-results");

  searchButton.addEventListener("click", function () {
    searchBox.classList.toggle("active");
    searchButton.classList.toggle("active");

    if (searchBox.classList.contains("active")) {
      searchInput.focus();
    } else {
      searchInput.value = ""; // Clear the search input when the box is hidden
      searchResults.innerHTML = ""; // Clear the search results when the box is hidden
    }
  });

  searchInput.addEventListener("input", function () {
    const searchTerm = searchInput.value.toLowerCase().trim();
    const allItems = document.querySelectorAll("your-item-selector"); // Replace with your item selector

    // Clear previous results
    searchResults.innerHTML = "";

    if (searchTerm === "") return;

    // Find the closest searched item
    const closestItems = Array.from(allItems).filter((item) => {
      return item.textContent.toLowerCase().includes(searchTerm);
    });

    closestItems.forEach((item) => {
      const listItem = document.createElement("li");
      listItem.textContent = item.textContent;
      listItem.addEventListener("click", function () {
        searchInput.value = item.textContent;
        searchResults.innerHTML = "";
      });
      searchResults.appendChild(listItem);
    });
  });
});
