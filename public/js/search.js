var searchElem = document.getElementById("search-input-page");
var posts = [];
var lastQuery = '';
var target = document.getElementById('list');
var postsByTitle = {};

function loadSearch() {
    var xhr = new XMLHttpRequest();
    var url = "../index.json";
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                var data = JSON.parse(xhr.responseText);
                if (data) {
                    posts = data;
                    postsByTitle = posts.reduce((acc, curr) => {
                        acc[curr.title] = curr;
                        return acc;
                    }, {});
                }
            } else {
                console.log(xhr.responseText);
            }
        }
    };
    xhr.open('GET', url);
    xhr.send();
}
loadSearch();

function formatDate(dateString) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    const date = new Date(dateString);
    const day = date.getDate();
    const month = date.toLocaleDateString(undefined, { month: 'long' });
    const year = date.getFullYear();

    const getDaySuffix = (day) => {
        if (day >= 11 && day <= 13) {
            return 'th';
        }
        switch (day % 10) {
            case 1: return 'st';
            case 2: return 'nd';
            case 3: return 'rd';
            default: return 'th';
        }
    };

    return `${month} ${day}${getDaySuffix(day)}, ${year}`;
}

function mySearch(query) {
    query = query.toLowerCase();
    var results = [];

    posts.forEach(function (post) {
        var title = post.title.toLowerCase();
        var content = post.content.toLowerCase();
        var url = post.url.toLowerCase();

        if (title.includes(query) || content.includes(query) || url.includes(query)) {
            results.push(post);
        }
    });

    return results;
}

function showSearchResults() {
    var query = searchElem.value || '';
    var searchString = query.trim();

    clearTimeout(window.searchTimeout);
    window.searchTimeout = setTimeout(function () {
        if (searchString === lastQuery) {
            return;
        }
        lastQuery = searchString;

        if (searchString && searchString !== '') {
            var matchPosts = mySearch(searchString);

            if (matchPosts.length > 0) {
                target.innerHTML = matchPosts.map(function (p) {
                    if (p !== undefined) {
                        return `<li>
                            ${formatDate(p.date)} -
                            <a href="${p.url}"> ${p.title}</a>
                            </li>`;
                    }
                }).join('');
            } else {
                target.innerHTML = `<br><h2 style="text-align:center">No search results found</h2>`;
            }
        } else {
            target.innerHTML = '';
        }
    }, 300);
}


searchElem.addEventListener('input', showSearchResults);
