(() => {
const form=document.querySelector("[data-search-form]");
const input=document.querySelector("[data-search-input]");
const results=document.querySelector("[data-search-results]");
const items=[...document.querySelectorAll("[data-search-item]")];
if(form&&input&&results){form.addEventListener("submit",event=>{event.preventDefault();const term=input.value.trim().toLowerCase();if(!term){results.classList.remove("is-visible");results.replaceChildren();return;}const matches=items.filter(item=>item.textContent.toLowerCase().includes(term));results.replaceChildren();const heading=document.createElement("strong");heading.textContent=matches.length?"Resultados para "+input.value.trim():"No hay resultados para esa búsqueda.";results.appendChild(heading);if(matches.length){const list=document.createElement("ul");matches.forEach(item=>{const source=item.querySelector("a");if(!source)return;const li=document.createElement("li");const link=document.createElement("a");link.href=source.href;link.textContent=source.textContent.trim();li.appendChild(link);list.appendChild(li);});results.appendChild(list);}results.classList.add("is-visible");});}
document.querySelectorAll("[data-year]").forEach(node=>node.textContent=String(new Date().getFullYear()));
})();