// Sample publication data
        const publications = [
            {
                title: "Vignette detection and reconstruction of composed ornaments with a strengthened autoencoder",
                authors: "Mohammad Sadil Khan , Rémi Emonet , Thierry Fournel",
                conference: "HAL Priprint",
                year: 2021,
                abstract: "A strengthened autoencoder formed by placing an object detector upstream of a decoder is here developed in the context of the model-helped human analysis of composed ornaments from a dictionary of vignettes. The detection part is in charge to detect regions of interest containing some vignette features, and the decoding part to ensure vignette reconstruction with a relative quality depending on feature match. Images of ornaments without typographical composition are generated in order to properly assess the performance of each of the two parts.",
                paperLink: "https://hal.science/hal-03409930"
            },
            {
                title: "Learning Shapes for Efficient Segmentation of 3D Medical Images using Point Cloud",
                authors: "Mohammad Sadil Khan, Razmig Kéchichian, Sebastien Valette, Julie Digne",
                conference: "Master Thesis",
                year: 2022,
                codeLink: "https://github.com/SadilKhan/Creatis-Internship",
                abstract: "In this report, we present a novel approach for 3D medical image segmentation using point clouds. 3D Convolutional Neural Networks have been the most dominating networks in medical image processing but they require large memory footprints and training samples. Hence we used point clouds to represent the image instead of voxels. Point clouds are lightweight and contain shape and smoother surface information. We extracted the point clouds from 3D voxel images using canny edge detection. We modified RandLa-Net, an attention-based point cloud segmentation network with a feature extraction layer to aggregate local geometrical features with spatial point features for our large-scale point cloud segmentation task. Our proposed model performed better than the original network in multi-class as well as binary point cloud segmentation tasks in Visceral dataset. Finally, we propose a model-independent step to perform the image segmentation of the original 3D volumetric images in Visceral dataset by mapping voxels in the point cloud space and adding it to the input point cloud before being passed to the trained model. We performed many experiments on the weights of the Cross-Entropy loss function for the class imbalance problem as well as the intrinsic architectural properties of the model architecture like downsampling factor and distinct latent vector learning that can be improved to perform better segmentation.",
                paperLink: "data/report.pdf"
            }
            // Add more publication objects here
        ];
        
        // Function to toggle year content visibility
        function toggleYearContent(year) {
        const yearContent = document.querySelector(`.year-content[data-year="${year}"]`);
        const arrowIcon = yearContent.previousElementSibling.querySelector('.arrow-icon');

        yearContent.classList.toggle('show'); // Toggle the 'show' class

        if (yearContent.classList.contains('show')) {
            arrowIcon.textContent = '▲';
        } else {
            arrowIcon.textContent = '▼';
        }
    }
let activeTooltip = null;

// Function to create and show the mini pop-up window
function showTooltip(event, content) {
    // Create the tooltip if it doesn't exist
    if (!activeTooltip) {
        activeTooltip = document.createElement('div');
        activeTooltip.classList.add('tooltip');
        document.body.appendChild(activeTooltip);

        // Listen for mouseleave event on the publication content
        const publicationContent = event.currentTarget;
        publicationContent.addEventListener('mouseleave', () => {
            document.body.removeChild(activeTooltip);
            activeTooltip = null;
        });
    }

    // Update the tooltip content and position
    activeTooltip.innerHTML = `<p><b> Abstract: <br> </b> ${content}</p>`;

    const xOffset = 10; // Adjust this value to set the horizontal offset
    const yOffset = -80; // Adjust this value to set the vertical offset
    const tooltipWidth = Math.min(600, activeTooltip.scrollWidth); // Set a maximum width of 400px
    const tooltipHeight = activeTooltip.scrollHeight;
    const adjustedTop = event.clientY - tooltipHeight - yOffset + window.scrollY;
const adjustedLeft = event.clientX + xOffset + tooltipWidth > window.innerWidth
        ? event.clientX - tooltipWidth + window.scrollX
        : event.clientX + xOffset;

    activeTooltip.style.maxWidth = `${tooltipWidth}px`;
    activeTooltip.style.left = `${adjustedLeft}px`;
    activeTooltip.style.top = `${adjustedTop}px`;
}




function displayPublications() {
            const publicationList = document.getElementById('publication-list');
            publicationList.innerHTML = ''; // Clear previous content

            // Sort publications by year in descending order (latest first)
            const sortedPublications = publications.sort((a, b) => b.year - a.year);

            let currentYear = null;

            let paperCount = 0; // Initialize paper count
            sortedPublications.forEach(publication => {
                if (publication.year !== currentYear) {
                    // Create a new section for each year
                    const yearSection = document.createElement('div');
                    yearSection.classList.add('year-section');
                    yearSection.innerHTML = `<h2 onclick="toggleYearContent(${publication.year})">
    ${publication.year} <span class="arrow-icon">▼</span>
</h2>
`;                  publicationList.appendChild(yearSection);

                    const yearContent = document.createElement('div');
                    yearContent.classList.add('year-content');
                    yearContent.setAttribute('data-year', publication.year);
                    yearSection.appendChild(yearContent);

                    currentYear = publication.year;
                    paperCount = 1; // Reset paper count for the new year
                }
                else {
                paperCount++; // Increment paper count for the same year
                }

                const publicationContent = document.createElement('div');
                publicationContent.classList.add('publication');

                publicationContent.innerHTML = `
                    <div class="paper-info">
                        <div class="paper-title"><b>${publication.title}</b></div>
                        <div class="paper-authors">${publication.authors}</div>
                        <div class="paper-conference"><i>${publication.conference}</i></div>
                    </div>
                    <div class="links">
                        ${publication.codeLink ? `<a href="${publication.codeLink}" target="_blank">Code</a>` : ''}
                        ${publication.paperLink ? `<a href="${publication.paperLink}" target="_blank">Paper</a>` : ''}
                    </div>
                `;
                // Add event listener to show mini pop-up on hover
        publicationContent.addEventListener('mouseover', event => {
            const tooltipContent = `
                ${publication.abstract}
            `;
            showTooltip(event, tooltipContent);
        });

                const yearContent = publicationList.querySelector(`.year-content[data-year="${publication.year}"]`);
                yearContent.appendChild(publicationContent);
            });
            
            
        }
  // Function to filter publications based on search input
        function filterPublications(searchText) {
    const filteredPublications = publications.filter(publication =>
        publication.title.toLowerCase().includes(searchText.toLowerCase()) ||
        publication.authors.toLowerCase().includes(searchText.toLowerCase()) ||
        publication.year.toString().includes(searchText) ||
        publication.conference.toLowerCase().includes(searchText.toLowerCase()) ||
        publication.abstract.toLowerCase().includes(searchText.toLowerCase())
    );
    return filteredPublications;
}

        // Function to update the displayed publications
        function updatePublications() {
            const searchText = document.getElementById('search').value;
            const filteredPublications = filterPublications(searchText);
            displayFilteredPublications(filteredPublications);
        }
        // Function to display filtered publications
        function displayFilteredPublications(filteredPublications) {
            const publicationList = document.getElementById('publication-list');
            publicationList.innerHTML = ''; // Clear previous content

            if (filteredPublications.length === 0) {
                publicationList.innerHTML = '<p>No matching publications found.</p>';
            } else {
            displayPublications();
            }
        }
        // Initial display of publications
        displayPublications();

        // Attach event listener to the search input
        document.getElementById('search').addEventListener('input', updatePublications);
