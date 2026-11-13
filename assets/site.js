(() => {
const lang=document.documentElement.lang||"es";
const forms=[...document.querySelectorAll("[data-search-form]")];
let promise;
const load=()=>promise||(promise=fetch("/search-index.json").then(r=>{if(!r.ok)throw new Error();return r.json();}).then(x=>x.articles||[]));
for(const form of forms){
 const input=form.querySelector("[data-search-input]");
 const results=document.querySelector("[data-search-results]");
 if(!input||!results)continue;
 form.addEventListener("submit",async e=>{
  e.preventDefault();const locale=form.dataset.searchLang||lang,term=input.value.trim().toLocaleLowerCase(locale);
  results.replaceChildren();if(!term){results.classList.remove("is-visible");return;}
  try{
   const all=await load(),hits=all.filter(x=>x.lang===locale&&(x.title+" "+x.summary+" "+x.labels).toLocaleLowerCase(locale).includes(term));
   const h=document.createElement("strong");h.textContent=hits.length?(locale==="en"?"Search results":"Resultados de búsqueda"):(locale==="en"?"No results found":"No se han encontrado resultados");results.appendChild(h);
   if(hits.length){const ul=document.createElement("ul");hits.forEach(x=>{const li=document.createElement("li"),a=document.createElement("a"),p=document.createElement("p");a.href=x.url;a.textContent=x.title;p.textContent=x.summary;li.append(a,p);ul.appendChild(li);});results.appendChild(ul);}
   results.classList.add("is-visible");
  }catch{const p=document.createElement("p");p.textContent=locale==="en"?"Search unavailable.":"Búsqueda no disponible.";results.appendChild(p);results.classList.add("is-visible");}
 });
}
const cards=[...document.querySelectorAll("[data-archive-card]")];
if(cards.length){const topic=new URLSearchParams(location.search).get("topic");let count=0;cards.forEach(c=>{const show=!topic||(c.dataset.topic||"").split(/\s+/).includes(topic);c.hidden=!show;if(show)count++;});const empty=document.querySelector("#archive-empty");if(empty)empty.hidden=count>0;}
document.querySelectorAll("[data-year]").forEach(n=>n.textContent=String(new Date().getFullYear()));
})();