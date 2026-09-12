"use strict";

const ROUTE_META = {
  shortest: {label: "最短距离", color: "#315f9b"},
  shade: {label: "遮荫优先", color: "#176b55"},
  utci: {label: "UTCI优先", color: "#c84e3b"}
};
const WARNING_TEXT = {
  MULTIDATE_HOT_WEATHER_SCENARIO: "当前采用多日期模型的2024-08-04极端高温晴天路径情景；地图阴影几何仍为2024-07-29。",
  FALLBACK_USED: "路线使用了空间Fallback路段。",
  PRIOR_IMPUTED_USED: "路线包含高不确定性的prior-imputed路段。",
  OUTSIDE_CENTER: "路线部分超出中心城区。",
  HIGH_UNCERTAINTY: "路线预测不确定性偏高。",
  LOW_RELIABILITY: "路线的模型可靠度偏低，请结合低可靠路段长度谨慎参考。"
};
const OBJECTIVE_HELP = {
  shortest: "最短距离：以总路程最短为首要目标，不主动为遮荫或热舒适增加绕行。",
  shade: "遮荫优先：在最大绕行率允许的范围内最大化遮荫比例；它通常会降低Tmrt，但并不直接以最低Tmrt作为成本目标。",
  utci: "UTCI热舒适优先：优先降低沿途预测热应激，适合关注整体热舒适体验。"
};
const state = {
  origin: null, destination: null, routes: [], selectedRouteId: null,
  visible: new Set(), context: null, fitRoutesOnly: false, mapTransform: null,
  mapPickTarget: null, visualLayers: null, buildingImage: null, shadowImage: null,
  visualTile: null, overviewVisualTile: null, visualLayerCache: new Map(),
  visualRefreshTimer: null,
  visualRequestToken: 0, mapZoom: 1, mapPanX: 0, mapPanY: 0,
  mapPointer: null, mapDragged: false, mapDrawFrame: null, plannerDrawFrame: null,
  plannerCases: [], plannerPoints: [],
  selectedPlannerPoint: null, plannerMapTransform: null, plannerPointLoading: false,
  plannerMapZoom: 1, plannerMapPanX: 0, plannerMapPanY: 0,
  plannerMapPointer: null, plannerMapDragged: false, plannerOriginalImageUrl: null,
  plannerVisualTile: null, plannerOverviewVisualTile: null,
  plannerVisualCache: new Map(), plannerVisualTimer: null, plannerVisualToken: 0,
  plannerOverlayToken: 0, plannerConfidenceTimer: null,
  plannerDirections: [], plannerDirectionResults: new Map(), plannerDirectionHeading: null,
  plannerDirectionBatch: null, plannerViewMode: "direction", plannerMapLayer: "plan",
  plannerOutputMode: "guidance", plannerGuidanceImage: null, plannerRealisticImage: null,
  plannerRealisticResult: null, plannerSource: "existing", plannerCurrentAnalysis: null,
  plannerMaskBaseImage: null, plannerMaskLayer: null, plannerMaskAutomatic: null,
  plannerMaskHistory: [], plannerMaskMode: "add", plannerMaskDrawing: false,
  plannerMaskDirty: false, plannerCommittedMask: null, plannerMaskLastPoint: null,
  plannerNoInterventionSubmitted: false, plannerNoInterventionReason: "",
  plannerReviewScope: "direction", plannerProgressTimer: null, plannerProgressStartedAt: 0,
  plannerProgressStages: [], plannerProgressStageIndex: 0
};
const $ = id => document.getElementById(id);
const MISSING_LABELS = {
  origin_query: "起点", destination_query: "终点", hour: "出发时间",
  mode: "交通模式", objective: "优化目标"
};

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? {"Content-Type": "application/json"} : {},
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store"
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || "本地服务请求失败");
  return data;
}

function beginPlannerProgress(title,stages,expectedSeconds=60){
  clearInterval(state.plannerProgressTimer);state.plannerProgressStartedAt=Date.now();
  state.plannerProgressStages=stages;state.plannerProgressStageIndex=0;
  const panel=$("planner-task-progress"),bar=$("planner-task-progress-bar"),value=$("planner-task-progress-value");
  panel.hidden=false;panel.className="planner-task-progress";$("planner-task-progress-title").textContent=title;
  const render=()=>{const elapsed=Math.max(0,(Date.now()-state.plannerProgressStartedAt)/1000),ratio=Math.min(.92,elapsed/Math.max(1,expectedSeconds)),percent=Math.max(4,Math.round(4+ratio*88)),index=Math.min(stages.length-1,Math.floor(ratio*stages.length));state.plannerProgressStageIndex=index;bar.value=percent;value.textContent=`预计 ${percent}%`;$("planner-task-progress-detail").textContent=`${stages[index]} · 已用时 ${Math.floor(elapsed)}秒`;renderPlannerProgressSteps(index);};
  render();state.plannerProgressTimer=setInterval(render,500);
}

function renderPlannerProgressSteps(activeIndex,status="running"){
  const list=$("planner-agent-steps");if(!list)return;
  list.innerHTML=state.plannerProgressStages.map((stage,index)=>`<li data-index="${index+1}" class="${status==="error"&&index===activeIndex?"error":index<activeIndex||status==="complete"?"done":index===activeIndex?"active":""}">${escapePlannerText(stage)}</li>`).join("");
}

function finishPlannerProgress(detail="处理完成"){
  clearInterval(state.plannerProgressTimer);state.plannerProgressTimer=null;
  const panel=$("planner-task-progress");panel.hidden=false;panel.className="planner-task-progress complete";$("planner-task-progress-bar").value=100;$("planner-task-progress-value").textContent="100%";$("planner-task-progress-detail").textContent=detail;
  renderPlannerProgressSteps(state.plannerProgressStages.length,"complete");
}

function failPlannerProgress(detail){
  clearInterval(state.plannerProgressTimer);state.plannerProgressTimer=null;
  const panel=$("planner-task-progress");panel.hidden=false;panel.className="planner-task-progress error";$("planner-task-progress-value").textContent="未完成";$("planner-task-progress-detail").textContent=detail;
  renderPlannerProgressSteps(state.plannerProgressStageIndex,"error");
}

function initializeResizablePanels(){
  const shell=document.querySelector(".app-shell"),results=document.querySelector(".results-panel");
  const bind=(handle,target,property,min,max,reverse=false,initial)=>{
    if(!handle||!target)return;
    const set=value=>{const clamped=Math.max(min,Math.min(max,value));target.style.setProperty(property,`${clamped}px`);localStorage.setItem(`shade-ui-${property}`,String(clamped));drawMap()};
    const saved=Number(localStorage.getItem(`shade-ui-${property}`));if(Number.isFinite(saved)&&saved>0)set(saved);
    const start=event=>{if(window.innerWidth<=980)return;event.preventDefault();handle.classList.add("dragging");handle.setPointerCapture(event.pointerId);const x=event.clientX,current=parseFloat(getComputedStyle(target).getPropertyValue(property))||initial;const move=e=>set(current+(e.clientX-x)*(reverse?-1:1));const end=()=>{handle.classList.remove("dragging");handle.removeEventListener("pointermove",move);handle.removeEventListener("pointerup",end);handle.removeEventListener("pointercancel",end)};handle.addEventListener("pointermove",move);handle.addEventListener("pointerup",end);handle.addEventListener("pointercancel",end)};
    handle.addEventListener("pointerdown",start);handle.addEventListener("dblclick",()=>set(initial));handle.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight","Home"].includes(event.key))return;event.preventDefault();if(event.key==="Home")return set(initial);const current=parseFloat(getComputedStyle(target).getPropertyValue(property))||initial;set(current+(event.key==="ArrowRight"?10:-10)*(reverse?-1:1))});
  };
  bind($("control-resizer"),shell,"--control-width",340,560,false,420);
  bind($("map-resizer"),results,"--compare-width",220,420,true,280);
}

function parsePlannerIntent(){
  const raw=$("planner-intent").value.trim();
  const objective=/遮阳棚|人工遮阳|雨棚/.test(raw)?"人工遮阳补充并优先保护慢行空间":/保留|不要移除|现有树/.test(raw)?"保留现有树木并补足连续树冠缺口":"树木优先补足步行骑行空间遮荫";
  const side=/南侧/.test(raw)?"道路南侧":/北侧/.test(raw)?"道路北侧":/左侧/.test(raw)?"画面左侧":/右侧/.test(raw)?"画面右侧":"道路两侧";
  const box=$("planner-intent-confirmation");box.hidden=false;
  $("planner-intent-objective").textContent=`目标：${objective}`;
  $("planner-intent-constraints").textContent=`范围：${side}。约束：避开道路尽头与建筑主体，遵循透视，仅编辑候选树冠/设施区域。${raw?` 原始需求：“${raw.slice(0,120)}”`:" 未填写额外约束，采用南京默认树木优先规则。"}`;
  $("analyze-planner-upload").focus();
}

function coordinate(candidate) {
  if (Number.isFinite(candidate.x) && Number.isFinite(candidate.y) && !Number.isFinite(candidate.longitude)) {
    return {crs: "EPSG:32650", x: candidate.x, y: candidate.y};
  }
  return {crs: "EPSG:4326", longitude: candidate.longitude, latitude: candidate.latitude};
}

async function searchPlace(kind) {
  const query = $(`${kind}-query`).value.trim();
  const box = $(`${kind}-candidates`);
  box.innerHTML = "<div class='candidate'>正在查询本地词典…</div>";
  try {
    const result = await api("/api/geocode", {query, limit: 10});
    box.innerHTML = "";
    if (!result.candidates.length) {
      box.innerHTML = "<div class='candidate'>未找到地名，请输入更具体名称。</div>";
      showWarnings(["离线OSM词典没有找到该名称，可改用道路、站点或地标名称，或直接在地图上点选。"]);
      return;
    }
    result.candidates.forEach(item => {
      const button = document.createElement("button");
      button.className = "candidate";
      button.innerHTML = `${item.display_name}<small>${item.feature_type} · 匹配 ${(item.match_score*100).toFixed(0)}%</small>`;
      button.onclick = () => {
        [...box.children].forEach(el => el.classList.remove("selected"));
        button.classList.add("selected");
        state[kind] = item;
        validateSnap(false);
        drawMap();
      };
      box.appendChild(button);
    });
    if (result.ambiguity_flag) showWarnings(["地名存在多个合理候选，请明确选择。"]);
  } catch (error) {
    box.innerHTML = `<div class="candidate">${error.message}</div>`;
  }
}

async function parseAssistantRequest() {
  const text=$("assistant-input").value.trim();
  const button=$("assistant-parse"), message=$("assistant-message");
  if(!text){message.className="assistant-message error";message.textContent="请先输入一句出行需求。";return}
  button.disabled=true;button.textContent="本地模型解析中…";
  message.className="assistant-message";message.textContent="Qwen3 正在本机 GPU 上解析，不会把内容发送到外部服务。";
  try{
    const result=await api("/api/assistant/parse",{text});
    if(result.origin_query)$("origin-query").value=result.origin_query;
    if(result.destination_query)$("destination-query").value=result.destination_query;
    if(result.hour!==null)$("hour").value=String(result.hour);
    if(result.mode)$("mode").value=result.mode;
    if(result.objective){
      const objective=ROUTE_META[result.objective]?result.objective:"shortest";
      $("objective").value=objective;
      $("objective-help").textContent=OBJECTIVE_HELP[objective];
    }
    if(result.max_detour_ratio!==null){
      $("detour").value=String(Math.round(result.max_detour_ratio*100));
      $("detour-value").textContent=`${$("detour").value}%`;
    }
    const searches=[];
    if(result.origin_query)searches.push(searchPlace("origin"));
    if(result.destination_query)searches.push(searchPlace("destination"));
    await Promise.all(searches);
    const missing=result.missing_fields.map(field=>MISSING_LABELS[field]||field);
    message.className=`assistant-message ${missing.length?"":"success"}`;
    message.textContent=missing.length
      ?`已填入可识别参数；还需手动设置：${missing.join("、")}。请明确选择起终点候选，再点击路线按钮确认。`
      :"解析完成。请分别选择起终点候选；路线不会自动计算，点击下方按钮即为确认。";
  }catch(error){
    message.className="assistant-message error";message.textContent=error.message;
  }finally{button.disabled=false;button.textContent="解析需求"}
}

async function explainRoutes() {
  const button=$("assistant-explain"), box=$("assistant-explanation");
  if(!state.routes.length)return;
  button.disabled=true;box.textContent="本地模型正在解释汇总指标…";
  const routes=state.routes.map(route=>({
    objective:route.objective,distance_m:route.distance_m,
    estimated_duration_min:route.estimated_duration_min,detour_ratio:route.detour_ratio,
    mean_shade:route.mean_shade,mean_tmrt:route.mean_tmrt,mean_utci:route.mean_utci,
    uncertainty_mean:route.uncertainty_mean,reliability_score:route.reliability_score,
    warning_codes:route.warning_codes||[]
  }));
  try{
    const result=await api("/api/assistant/explain",{routes,question:"请解释这些路线的距离、遮荫、热舒适与不确定性权衡。"});
    box.textContent=result.explanation;
  }catch(error){box.textContent=error.message}
  finally{button.disabled=false}
}

function routePayload() {
  if (!state.origin || !state.destination) throw new Error("请先从候选列表选择起点和终点。");
  return {
    origin: coordinate(state.origin),
    destination: coordinate(state.destination),
    hour: Number($("hour").value),
    mode: $("mode").value,
    objective: $("objective").value,
    algorithm: "astar",
    max_detour_ratio: Number($("detour").value) / 100,
    uncertainty_weight: 0
  };
}

function setRouteLoading(active,compare=false){
  ["compare-routes","single-route"].forEach(id=>{
    const button=$(id);if(!button.dataset.idleText)button.dataset.idleText=button.textContent;
    button.disabled=active;button.setAttribute("aria-busy",active?"true":"false");
    button.textContent=active&&((compare&&id==="compare-routes")||(!compare&&id==="single-route"))?"正在计算路线…":button.dataset.idleText;
  });
}

async function calculate(compare) {
  setRouteLoading(true,compare);
  $("map-message").textContent = "正在使用本地确定性引擎计算…";
  $("map-message").hidden = false;
  try {
    if (!(await validateSnap(true))) return;
    const payload = routePayload();
    const result = await api(compare ? "/api/compare" : "/api/route",
      compare ? {...payload, objectives: Object.keys(ROUTE_META)} : payload);
    state.routes = compare ? result.routes : [result];
    state.visible = new Set(state.routes.map(route => route.route_id));
    state.selectedRouteId = state.routes[0]?.route_id || null;
    state.fitRoutesOnly = false;
    state.mapZoom = 1; state.mapPanX = 0; state.mapPanY = 0;
    renderResults();
    $("assistant-explain-wrap").hidden=false;
    $("assistant-explanation").textContent="";
    $("map-title").textContent = `${$("hour").value}:00 · ${$("mode").selectedOptions[0].text}`;
    $("map-message").hidden = true;
    loadVisualLayers();
  } catch (error) {
    $("map-message").textContent = error.message;
    showWarnings([error.message]);
  } finally { setRouteLoading(false,compare); }
}

async function validateSnap(showSuccess) {
  if (!state.origin || !state.destination) return false;
  const box=$("snap-status");
  box.className="snap-status"; box.textContent="正在检查100米吸附与连通性…";
  try {
    const result=await api("/api/snap",{
      origin:coordinate(state.origin),destination:coordinate(state.destination),mode:$("mode").value
    });
    box.className="snap-status ok";
    box.textContent=`吸附通过：起点 ${fmt(result.origin_snap_distance_m)} m，终点 ${fmt(result.destination_snap_distance_m)} m，位于同一连通分量。`;
    if(showSuccess) showWarnings([]);
    return true;
  } catch(error) {
    box.className="snap-status error";
    box.textContent=`${error.message} 请更换同名候选，或使用地图点选到附近道路。`;
    showWarnings([box.textContent]);
    return false;
  }
}

function fmt(value, digits=1) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}
function routeName(route) { return ROUTE_META[route.objective]?.label || route.objective; }
function reliabilityLabel(route) {
  const score=Number(route.reliability_score);
  const grade={high:"高",medium:"中",low:"低"}[route.reliability_grade]||"待定";
  return Number.isFinite(score)?`${score.toFixed(0)}/100（${grade}）`:"—";
}
function segmentOverlapWithShortest(route) {
  const shortest=state.routes.find(item=>item.objective==="shortest");
  if(!shortest || route.objective==="shortest") return route.objective==="shortest"?1:null;
  const base=new Set(shortest.segment_ids||[]),candidate=new Set(route.segment_ids||[]);
  if(!base.size&&!candidate.size)return null;
  let intersection=0;candidate.forEach(id=>{if(base.has(id))intersection++});
  return intersection/Math.max(1,new Set([...base,...candidate]).size);
}

function renderResults() {
  $("empty-results").hidden = state.routes.length > 0;
  $("comparison-table").hidden = state.routes.length === 0;
  $("route-metrics-details").hidden = state.routes.length === 0;
  const cards = $("route-cards");
  const tbody = $("comparison-table").querySelector("tbody");
  cards.innerHTML = ""; tbody.innerHTML = "";
  state.routes.forEach(route => {
    const objectiveBadge = {
      shortest: "目标：最短距离",
      shade: "目标：最高遮荫",
      utci: "目标：最低UTCI"
    }[route.objective];
    const overlap=segmentOverlapWithShortest(route);
    const card = document.createElement("button");
    card.className = `route-card ${route.route_id === state.selectedRouteId ? "selected" : ""}`;
    card.setAttribute("aria-pressed",route.route_id === state.selectedRouteId ? "true" : "false");
    card.innerHTML = `<strong style="color:${ROUTE_META[route.objective].color}">${routeName(route)}</strong>
      <span class="route-card-sub">${fmt(route.distance_m/1000,2)} km · 约 ${fmt(route.estimated_duration_min)} min</span>
      <div class="route-key-metrics">
        <span><b>${fmt(route.mean_shade*100,0)}%</b><small>平均遮荫</small></span>
        <span><b>${fmt(route.mean_utci,1)}°</b><small>平均UTCI</small></span>
      </div>
      <div class="route-card-foot"><i class="badge">${objectiveBadge}</i><span>可靠度 ${reliabilityLabel(route)}</span></div>`;
    card.onclick = () => selectRoute(route.route_id);
    cards.appendChild(card);
    const row = document.createElement("tr");
    row.className = route.route_id === state.selectedRouteId ? "active" : "";
    row.innerHTML = `<td><input type="checkbox" data-route="${route.route_id}" ${state.visible.has(route.route_id)?"checked":""}></td>
      <td>${routeName(route)}</td><td>${fmt(route.distance_m/1000,2)} km</td>
      <td>${fmt(route.detour_ratio*100)}%</td><td>${fmt(route.estimated_duration_min)} min</td>
      <td>${fmt(route.mean_shade*100)}%</td><td>${fmt(route.shade_exposure*100)}%</td>
      <td>${fmt(route.mean_tmrt)}°C</td><td>${fmt(route.max_tmrt)}°C</td>
      <td>${fmt(route.mean_utci)}°C</td><td>${fmt(route.max_utci)}°C</td>
      <td title="相对预测不确定性与数据覆盖综合评分，不是统计置信区间">${reliabilityLabel(route)}</td>
      <td>${overlap===null?"—":`${(overlap*100).toFixed(0)}%`}</td>
      <td>${fmt(route.low_reliability_length_m)} m</td>`;
    row.onclick = event => { if (event.target.type !== "checkbox") selectRoute(route.route_id); };
    tbody.appendChild(row);
  });
  tbody.querySelectorAll("input[type=checkbox]").forEach(box => {
    box.onchange = () => {
      box.checked ? state.visible.add(box.dataset.route) : state.visible.delete(box.dataset.route);
      drawMap();
    };
  });
  renderLegend();
  drawMap();
  const warnings = [...new Set(state.routes.flatMap(r => r.warning_codes).map(c => WARNING_TEXT[c]).filter(Boolean))];
  const uniqueSequences=new Set(state.routes.map(r=>(r.segment_ids||[]).join("|"))).size;
  if(state.routes.length>1 && uniqueSequences<state.routes.length){
    warnings.push(`当前三类目标只有 ${uniqueSequences} 条独立线路；其余结果在当前${$("detour").value}%绕行约束和路网成本下重合。遮荫与UTCI成本通常相关，系统不会为了视觉差异人为制造绕路。`);
  }
  showWarnings(warnings);
  if (state.selectedRouteId) loadSummary();
}

function selectRoute(routeId) {
  state.selectedRouteId = routeId;
  state.visible.add(routeId);
  renderResults();
}
async function loadSummary() {
  try {
    const result = await api("/api/summarize", {route_id: state.selectedRouteId, language: "zh-CN"});
    $("summary-box").hidden = false;
    $("summary-box").textContent = `${result.summary} ${result.scenario_disclosure}`;
  } catch (_) { $("summary-box").hidden = true; }
}

function renderLegend() {
  const legend=$("route-legend");
  legend.innerHTML="";
  state.routes.forEach(route=>{
    const button=document.createElement("button");
    button.className=`legend-item ${route.route_id===state.selectedRouteId?"selected":""}`;
    button.setAttribute("aria-pressed",route.route_id===state.selectedRouteId?"true":"false");
    button.style.setProperty("--route-color",ROUTE_META[route.objective].color);
    button.innerHTML=`<span class="legend-line" style="background:${ROUTE_META[route.objective].color}"></span>${routeName(route)}`;
    button.title=`点击突出显示${routeName(route)}`;
    button.onclick=()=>selectRoute(route.route_id);
    legend.appendChild(button);
  });
}

function allCoordinates() {
  const routes = state.routes.filter(r => state.visible.has(r.route_id))
    .flatMap(r => r.geometry?.coordinates || []);
  if (state.fitRoutesOnly && routes.length) return routes;
  const bbox=state.context?.bbox;
  if(routes.length){
    const xs=routes.map(point=>point[0]),ys=routes.map(point=>point[1]);
    const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const span=Math.max(maxX-minX,maxY-minY),pad=Math.max(1800,span*.65);
    const view=[
      bbox?Math.max(bbox[0],minX-pad):minX-pad,
      bbox?Math.max(bbox[1],minY-pad):minY-pad,
      bbox?Math.min(bbox[2],maxX+pad):maxX+pad,
      bbox?Math.min(bbox[3],maxY+pad):maxY+pad
    ];
    return routes.concat([[view[0],view[1]],[view[2],view[3]]]);
  }
  return bbox ? [[bbox[0],bbox[1]],[bbox[2],bbox[3]]] : routes;
}

function visualLayerRequest(explicitBbox=null) {
  const context=[...state.context.bbox];
  if(explicitBbox) return {bbox:explicitBbox,rasterMaxDimension:1200,isOverview:true};
  const visible=state.mapTransform?.viewExtent||context;
  const spanX=Math.max(1,visible[2]-visible[0]),spanY=Math.max(1,visible[3]-visible[1]);
  const contextSpan=Math.max(context[2]-context[0],context[3]-context[1]);
  if(Math.max(spanX,spanY)>=contextSpan*.78){
    return {bbox:context,rasterMaxDimension:1200,isOverview:true};
  }
  const padding=Math.max(80,Math.max(spanX,spanY)*.16);
  const raw=[
    Math.max(context[0],visible[0]-padding),Math.max(context[1],visible[1]-padding),
    Math.min(context[2],visible[2]+padding),Math.min(context[3],visible[3]+padding)
  ];
  const grid=Math.max(20,Math.max(spanX,spanY)/12);
  const bbox=[
    Math.max(context[0],Math.floor(raw[0]/grid)*grid),
    Math.max(context[1],Math.floor(raw[1]/grid)*grid),
    Math.min(context[2],Math.ceil(raw[2]/grid)*grid),
    Math.min(context[3],Math.ceil(raw[3]/grid)*grid)
  ];
  const rect=$("route-map").getBoundingClientRect();
  const rasterMaxDimension=Math.min(1400,Math.max(900,Math.round(Math.max(rect.width,rect.height)*(window.devicePixelRatio||1)*1.1)));
  return {bbox,rasterMaxDimension,isOverview:false};
}

function scheduleVisualLayerRefresh(force=false) {
  clearTimeout(state.visualRefreshTimer);
  state.visualRefreshTimer=setTimeout(()=>loadVisualLayers(force),360);
}

function tileCovers(tile,extent) {
  if(!tile?.layers?.bbox||!extent) return false;
  const b=tile.layers.bbox;
  return b[0]<=extent[0]&&b[1]<=extent[1]&&b[2]>=extent[2]&&b[3]>=extent[3];
}

async function loadVisualLayers(force=false,explicitBbox=null) {
  if(!state.context?.bbox) return;
  const request=visualLayerRequest(explicitBbox);
  const {bbox,rasterMaxDimension,isOverview}=request;
  const requestedHour=Number($("hour").value);
  const key=`${requestedHour}|${rasterMaxDimension}|${bbox.map(value=>Math.round(value)).join(",")}`;
  if(!force && state.visualTile?.key===key) return;
  if(!force && state.visualLayerCache.has(key)){
    state.visualTile=state.visualLayerCache.get(key);drawMap();return;
  }
  const token=++state.visualRequestToken;
  $("layer-status").className="layer-status loading";
  $("layer-status").textContent=isOverview
    ?`正在加载全域建筑与 ${String(requestedHour).padStart(2,"0")}:00 阴影…`
    :`正在按当前视野提高清晰度（${String(requestedHour).padStart(2,"0")}:00）…`;
  try {
    const layers=await api("/api/visual-layers",{
      bbox,hour:requestedHour,max_buildings:100,raster_max_dimension:rasterMaxDimension,include_vectors:false
    });
    if(token!==state.visualRequestToken) return;
    const loadImage=base64=>new Promise((resolve,reject)=>{
      const image=new Image();
      image.onload=()=>resolve(image); image.onerror=reject;
      image.src=`data:image/png;base64,${base64}`;
    });
    const [buildingImage,shadowImage]=await Promise.all([
      loadImage(layers.building_png_base64),loadImage(layers.shadow_png_base64)
    ]);
    if(token!==state.visualRequestToken) return;
    delete layers.building_png_base64;
    delete layers.shadow_png_base64;
    const tile={key,layers,buildingImage,shadowImage};
    state.visualTile=tile;
    if(isOverview) state.overviewVisualTile=tile;
    state.visualLayerCache.set(key,tile);
    while(state.visualLayerCache.size>8) state.visualLayerCache.delete(state.visualLayerCache.keys().next().value);
    state.visualLayers=layers;state.buildingImage=buildingImage;state.shadowImage=shadowImage;
    $("layer-status").className="layer-status ready";
    const metersPerPixel=Math.max(
      (layers.bbox[2]-layers.bbox[0])/layers.building_width,
      (layers.bbox[3]-layers.bbox[1])/layers.building_height
    );
    $("layer-status").textContent=
      `${isOverview?"全域":"当前视野"}建筑与阴影已加载 · 约 ${metersPerPixel.toFixed(metersPerPixel<10?1:0)} 米/像素`+
      ` · ${layers.building_raster_feature_count.toLocaleString()} 个建筑轮廓 · ${String(layers.hour).padStart(2,"0")}:00`;
    drawMap();
  } catch(error) {
    if(token!==state.visualRequestToken) return;
    $("layer-status").className="layer-status error";
    $("layer-status").textContent=`建筑与阴影图层加载失败：${error.message}；路线结果不受影响。`;
  }
}

function drawBuildings(ctx, point, scale, layers=state.visualLayers) {
  if(!layers?.buildings?.length) return;
  const detail=Math.min(1,Math.max(.28,scale/.10));
  layers.buildings.forEach(building=>{
    const lift=Math.min(7,Math.max(.8,building.height*scale*.12));
    const offset=[-lift*.62,-lift];
    building.rings.forEach(ring=>{
      const base=ring.map(point);
      if(base.length<4) return;
      ctx.fillStyle=`rgba(112,121,116,${(.08*detail).toFixed(3)})`;
      for(let i=0;i<base.length-1;i++){
        const a=base[i],b=base[i+1];
        ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);
        ctx.lineTo(b[0]+offset[0],b[1]+offset[1]);
        ctx.lineTo(a[0]+offset[0],a[1]+offset[1]);ctx.closePath();ctx.fill();
      }
      ctx.beginPath();
      base.forEach((p,index)=>index?ctx.lineTo(p[0]+offset[0],p[1]+offset[1]):ctx.moveTo(p[0]+offset[0],p[1]+offset[1]));
      ctx.closePath();ctx.fillStyle=`rgba(216,214,204,${(.30*detail).toFixed(3)})`;ctx.fill();
      ctx.strokeStyle=`rgba(112,121,116,${(.18*detail).toFixed(3)})`;ctx.lineWidth=.35;ctx.stroke();
    });
  });
}

function drawMap() {
  const canvas = $("route-map");
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = rect.width * scale; canvas.height = rect.height * scale;
  const ctx = canvas.getContext("2d"); ctx.scale(scale, scale);
  ctx.fillStyle = "#eef0ea"; ctx.fillRect(0, 0, rect.width, rect.height);
  const coords = allCoordinates();
  if (!coords.length) return;
  let minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity;
  coords.forEach(p=>{minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1])});
  const dataSpan=Math.max(maxX-minX,maxY-minY);
  const pad = state.fitRoutesOnly && state.routes.length
    ? Math.max(500,dataSpan*.12)
    : dataSpan*(state.routes.length ? .03 : .02) || 100;
  minX-=pad; maxX+=pad; minY-=pad; maxY+=pad;
  const baseS=Math.min((rect.width-30)/(maxX-minX),(rect.height-30)/(maxY-minY));
  const baseTx=(rect.width-(maxX-minX)*baseS)/2,baseTy=(rect.height-(maxY-minY)*baseS)/2;
  const centerX=rect.width/2,centerY=rect.height/2,zoom=state.mapZoom;
  const point=p=>{
    const bx=baseTx+(p[0]-minX)*baseS,by=rect.height-(baseTy+(p[1]-minY)*baseS);
    return [centerX+(bx-centerX)*zoom+state.mapPanX,centerY+(by-centerY)*zoom+state.mapPanY];
  };
  const unproject=(sx,sy)=>{
    const bx=centerX+(sx-centerX-state.mapPanX)/zoom;
    const by=centerY+(sy-centerY-state.mapPanY)/zoom;
    return [minX+(bx-baseTx)/baseS,minY+(rect.height-by-baseTy)/baseS];
  };
  const viewCorners=[unproject(0,0),unproject(rect.width,0),unproject(0,rect.height),unproject(rect.width,rect.height)];
  const viewMinX=Math.min(...viewCorners.map(p=>p[0])),viewMaxX=Math.max(...viewCorners.map(p=>p[0]));
  const viewMinY=Math.min(...viewCorners.map(p=>p[1])),viewMaxY=Math.max(...viewCorners.map(p=>p[1]));
  const viewExtent=[viewMinX,viewMinY,viewMaxX,viewMaxY];
  const s=baseS*zoom;
  const activeTile=tileCovers(state.visualTile,viewExtent)?state.visualTile:state.overviewVisualTile;
  const activeLayers=activeTile?.layers;
  const shadowsVisible=$("show-shadows")?.checked;
  const buildingsVisible=$("show-buildings")?.checked;
  if(shadowsVisible && activeTile?.shadowImage && activeLayers?.shadow_extent){
    const extent=activeLayers.shadow_extent;
    const topLeft=point([extent[0],extent[3]]),bottomRight=point([extent[2],extent[1]]);
    ctx.drawImage(activeTile.shadowImage,topLeft[0],topLeft[1],bottomRight[0]-topLeft[0],bottomRight[1]-topLeft[1]);
  }
  if(buildingsVisible && activeTile?.buildingImage && activeLayers?.building_extent){
    const extent=activeLayers.building_extent;
    const topLeft=point([extent[0],extent[3]]),bottomRight=point([extent[2],extent[1]]);
    ctx.drawImage(activeTile.buildingImage,topLeft[0],topLeft[1],bottomRight[0]-topLeft[0],bottomRight[1]-topLeft[1]);
  } else if(buildingsVisible) drawBuildings(ctx,point,s,activeLayers);
  if (state.context) {
    const labels=[];
    state.context.features.forEach(f => {
      const b=f._bbox;
      if(b && (b[2]<viewMinX || b[0]>viewMaxX || b[3]<viewMinY || b[1]>viewMaxY)) return;
      const boundary=f.properties?.layer==="center_boundary";
      const roadClass=f.properties?.road_class||"";
      const major=["motorway","trunk","primary","motorway_link","trunk_link","primary_link"].includes(roadClass);
      const secondary=["secondary","tertiary","secondary_link","tertiary_link"].includes(roadClass);
      const active=["footway","pedestrian","cycleway","path","steps"].includes(roadClass);
      const local=["residential","unclassified","living_street"].includes(roadClass);
      const service=roadClass==="service",connector=roadClass==="topology_connector";
      ctx.strokeStyle=boundary?"rgba(100,119,111,.72)":major?"rgba(184,159,116,.58)":secondary?"rgba(166,175,166,.68)":active?"rgba(120,153,135,.62)":local?"rgba(163,174,166,.58)":service?"rgba(177,185,179,.42)":"rgba(188,194,189,.30)";
      ctx.lineWidth=boundary?1.4:major?2.1:secondary?1.45:active?1.0:local?1.05:service?.7:.6;
      ctx.setLineDash(connector?[3,3]:[]);
      const lines=f.geometry.type==="LineString"?[f.geometry.coordinates]:f.geometry.coordinates;
      lines.forEach(line => { ctx.beginPath(); line.forEach((p,i)=>{const q=point(p); i?ctx.lineTo(...q):ctx.moveTo(...q)}); ctx.stroke(); });
      if(f.properties?.name && s>.10 && (major||secondary) && labels.length<35){
        const line=lines[0],mid=line[Math.floor(line.length/2)];
        if(mid) labels.push([point(mid),f.properties.name]);
      }
    });
    ctx.setLineDash([]);
    ctx.font="10px 'Microsoft YaHei UI',sans-serif";ctx.textAlign="center";
    labels.forEach(([p,name])=>{
      ctx.lineWidth=3;ctx.strokeStyle="rgba(255,255,255,.9)";ctx.strokeText(name,p[0],p[1]);
      ctx.fillStyle="#6f685b";ctx.fillText(name,p[0],p[1]);
    });
  }
  const visibleRoutes=state.routes.filter(r => state.visible.has(r.route_id))
    .sort((a,b)=>(a.route_id===state.selectedRouteId?1:0)-(b.route_id===state.selectedRouteId?1:0));
  visibleRoutes.forEach((route,index) => {
    const line=route.geometry?.coordinates || []; if (!line.length) return;
    const dashes={shortest:[],shade:[11,6],utci:[3,5]};
    ctx.setLineDash(dashes[route.objective]||[]);
    const selectedRoute=route.route_id===state.selectedRouteId;
    ctx.globalAlpha=selectedRoute?1:.58;
    ctx.strokeStyle=ROUTE_META[route.objective].color; ctx.lineWidth=selectedRoute?3.4:1.5; ctx.lineCap="round"; ctx.lineJoin="round";
    ctx.beginPath(); line.forEach((p,i)=>{const q=point(p); i?ctx.lineTo(...q):ctx.moveTo(...q)}); ctx.stroke();
  });
  ctx.globalAlpha=1;
  ctx.setLineDash([]);
  const selected=state.routes.find(r=>r.route_id===state.selectedRouteId);
  if (selected?.geometry?.coordinates?.length) {
    const line=selected.geometry.coordinates, a=point(line[0]), b=point(line.at(-1));
    [[a,"#176b55","起"],[b,"#c84e3b","终"]].forEach(([p,c,t])=>{ctx.fillStyle=c;ctx.beginPath();ctx.arc(p[0],p[1],8,0,Math.PI*2);ctx.fill();ctx.fillStyle="#fff";ctx.font="bold 10px sans-serif";ctx.textAlign="center";ctx.fillText(t,p[0],p[1]+3)});
  }
  if(!selected){
    [["origin","#176b55","起"],["destination","#c84e3b","终"]].forEach(([kind,color,label])=>{
      const value=state[kind];
      if(!value || !Number.isFinite(value.x) || !Number.isFinite(value.y)) return;
      const p=point([value.x,value.y]);
      ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],8,0,Math.PI*2);ctx.fill();
      ctx.fillStyle="#fff";ctx.font="bold 10px sans-serif";ctx.textAlign="center";ctx.fillText(label,p[0],p[1]+3);
    });
  }
  const meters = niceScale(110/s);
  $("scale-bar").textContent = meters >= 1000 ? `${meters/1000} km` : `${meters} m`;
  $("scale-bar").style.width = `${meters*s}px`;
  state.mapTransform={project:point,unproject,width:rect.width,height:rect.height,pixelsPerMeter:s,viewExtent};
}
function niceScale(value) {
  const power=10**Math.floor(Math.log10(value)); const n=value/power;
  return (n<2?1:n<5?2:5)*power;
}
function zoomMap(multiplier,clientX=null,clientY=null) {
  const canvas=$("route-map"),rect=canvas.getBoundingClientRect(),old=state.mapZoom;
  const next=Math.min(12,Math.max(.65,old*multiplier));if(next===old)return;
  const sx=clientX===null?rect.width/2:clientX-rect.left,sy=clientY===null?rect.height/2:clientY-rect.top;
  const cx=rect.width/2,cy=rect.height/2,k=next/old;
  state.mapPanX=(sx-cx)-(sx-cx-state.mapPanX)*k;
  state.mapPanY=(sy-cy)-(sy-cy-state.mapPanY)*k;
  state.mapZoom=next;
  drawMap();
  scheduleVisualLayerRefresh();
}
function beginMapDrag(event){
  if(event.button!==undefined&&event.button!==0)return;
  state.mapPointer={id:event.pointerId,x:event.clientX,y:event.clientY,panX:state.mapPanX,panY:state.mapPanY};
  state.mapDragged=false;try{event.currentTarget.setPointerCapture?.(event.pointerId)}catch{}
  event.currentTarget.classList.add("dragging");
}
function moveMap(event){
  const drag=state.mapPointer;if(!drag||drag.id!==event.pointerId)return;
  const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
  if(Math.abs(dx)+Math.abs(dy)>4)state.mapDragged=true;
  state.mapPanX=drag.panX+dx;state.mapPanY=drag.panY+dy;
  if(!state.mapDrawFrame)state.mapDrawFrame=requestAnimationFrame(()=>{state.mapDrawFrame=null;drawMap()});
}
function endMapDrag(event){
  if(!state.mapPointer||state.mapPointer.id!==event.pointerId)return;
  try{event.currentTarget.releasePointerCapture?.(event.pointerId)}catch{}
  event.currentTarget.classList.remove("dragging");state.mapPointer=null;
  scheduleVisualLayerRefresh();
}
function handleMapKey(event){
  const step=event.shiftKey?90:45;
  if(event.key==="ArrowLeft")state.mapPanX+=step;
  else if(event.key==="ArrowRight")state.mapPanX-=step;
  else if(event.key==="ArrowUp")state.mapPanY+=step;
  else if(event.key==="ArrowDown")state.mapPanY-=step;
  else if(event.key==="+"||event.key==="="){zoomMap(1.25);event.preventDefault();return}
  else if(event.key==="-"||event.key==="_"){zoomMap(1/1.25);event.preventDefault();return}
  else if(event.key==="Escape"&&state.mapPickTarget){state.mapPickTarget=null;document.querySelectorAll(".map-pick-button").forEach(button=>button.classList.remove("active"));$("route-map").classList.remove("picking");}
  else return;
  event.preventDefault();drawMap();scheduleVisualLayerRefresh();
}
function showWarnings(items) {
  const box=$("warnings"); box.hidden=!items.length;
  box.innerHTML=items.length?`<strong>提示</strong><ul>${items.map(x=>`<li>${x}</li>`).join("")}</ul>`:"";
}

async function exportSelected() {
  if (!state.selectedRouteId) return showWarnings(["请先选择一条路线。"]);
  try {
    const result=await api("/api/export",{route_id:state.selectedRouteId,format:$("export-format").value});
    if(result.download_url){const link=document.createElement("a"); link.href=result.download_url; link.download=result.filename; link.click()}
    showWarnings([`已生成本地导出：${result.filename}${result.download_url?"":"（文件夹保存在应用输出目录）"}`]);
  } catch(error) { showWarnings([error.message]); }
}

function clearRequest() {
  state.origin=null; state.destination=null; state.routes=[]; state.visible.clear(); state.selectedRouteId=null;
  state.mapPickTarget=null; state.fitRoutesOnly=false;
  state.mapZoom=1;state.mapPanX=0;state.mapPanY=0;
  ["origin-query","destination-query"].forEach(id=>$(id).value="");
  ["origin-candidates","destination-candidates","route-cards","route-legend"].forEach(id=>$(id).innerHTML="");
  $("comparison-table").querySelector("tbody").innerHTML="";
  $("comparison-table").hidden=true; $("empty-results").hidden=false; $("summary-box").hidden=true;
  $("route-metrics-details").hidden=true;
  $("assistant-explain-wrap").hidden=true;$("assistant-explanation").textContent="";
  $("map-title").textContent="在左侧设置行程";
  $("map-message").hidden=false; $("map-message").textContent="选择起点和终点后开始计算";
  $("snap-status").className="snap-status"; $("snap-status").textContent="选择起终点后将自动检查路网连接。";
  document.querySelectorAll(".map-pick-button").forEach(button=>button.classList.remove("active"));
  $("route-map").classList.remove("picking","dragging");
  showWarnings([]); drawMap(); scheduleVisualLayerRefresh();
}

function clearPlace(kind) {
  state[kind]=null;
  $(`${kind}-query`).value="";
  $(`${kind}-candidates`).innerHTML="";
  $(`${kind}-map-pick`).classList.remove("active");
  if(state.mapPickTarget===kind) state.mapPickTarget=null;
  $("route-map").classList.remove("picking","dragging");
  state.routes=[];state.visible.clear();state.selectedRouteId=null;
  state.fitRoutesOnly=false;state.mapZoom=1;state.mapPanX=0;state.mapPanY=0;
  $("route-cards").innerHTML="";$("route-legend").innerHTML="";
  $("comparison-table").querySelector("tbody").innerHTML="";
  $("comparison-table").hidden=true;$("empty-results").hidden=false;$("summary-box").hidden=true;
  $("route-metrics-details").hidden=true;
  $("assistant-explain-wrap").hidden=true;$("assistant-explanation").textContent="";
  $("map-title").textContent="在左侧设置行程";
  $("map-message").hidden=false;$("map-message").textContent="选择起点和终点后开始计算";
  $("snap-status").className="snap-status";
  $("snap-status").textContent="选择起终点后将自动检查路网连接。";
  showWarnings([]);drawMap();scheduleVisualLayerRefresh();
}

function switchWorkspace(name) {
  document.querySelectorAll(".workspace-tab").forEach(button => {
    const active=button.dataset.workspace === name;button.classList.toggle("active",active);button.setAttribute("aria-selected",active?"true":"false");
  });
  $("route-workspace").hidden = name !== "route";
  $("planner-workspace").hidden = name !== "planner";
  if (name === "route") setTimeout(drawMap, 0);
  if (name === "planner" && !state.plannerPoints.length) loadPlannerPoints();
  if (name === "planner") setTimeout(() => { drawPlannerMap(); schedulePlannerVisualRefresh(); }, 0);
}

function switchPlannerSource(name) {
  state.plannerSource=name;
  document.querySelectorAll(".planner-source").forEach(button => {
    const active=button.dataset.source === name;button.classList.toggle("active",active);button.setAttribute("aria-selected",active?"true":"false");
  });
  $("planner-existing-source").hidden = name !== "existing";
  $("planner-upload-source").hidden = name !== "upload";
  if(name==="upload")$("planner-direction-panel").hidden=true;
}

function metric(label, value) {
  return `<div class="planner-metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function escapePlannerText(value){return String(value??"").replace(/[&<>'"]/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));}

function plannerDecisionPipeline(item={}){
  const order=["stage_1_shade_need","stage_2_phenology_gate","stage_3_space_typology","stage_4_tree_evidence","stage_4_optimization","stage_5_optimization"];
  const labels=["遮荫现状","物候状态","空间类型","树木证据","工具执行"];
  const current=Math.max(0,order.indexOf(item.decision_stage));
  return `<div class="planner-stage-flow" aria-label="规划判断阶段">${labels.map((label,index)=>{
    const status=index<Math.min(current,4)?"done":index===Math.min(current,4)?"active":"skipped";
    return `<div class="planner-stage ${status}"><span>${index+1}</span><b>${label}</b><small>${status==="done"?"已完成":status==="active"?"当前结论":"按需调用"}</small></div>`;
  }).join("")}</div><p class="decision-status"><strong>流程状态：</strong>${item.decision_status||"等待判断"}</p>`;
}

function showOverlayLegend(legend={}) {
  const meta={canopy_target:["canopy-zone","树冠目标覆盖区域（上边界即目标高度）"],facility_review:["facility-zone","人工遮阳棚候选覆盖区域"],facility_anchor:["facility-anchor-zone","可依附围墙/杆件/路牌"],optimization_azimuth_lines:["shade-gap-zone","优化方位竖线"]};
  const box=$("planner-overlay-legend"),items=Object.keys(legend).filter(key=>meta[key]);
  box.hidden=!items.length;box.innerHTML=items.map(key=>`<span><i class="${meta[key][0]}"></i>${meta[key][1]}</span>`).join("");
}

function renderPublicSpaceMeter(score,label="概率判断"){
  const box=$("public-space-meter"),value=Math.max(0,Math.min(1,Number(score)));
  if(!Number.isFinite(value)){box.hidden=true;return}
  box.hidden=false;const percent=Math.round(value*100);$("public-space-score").textContent=`${percent}/100 · ${label}`;
  $("public-space-indicator").style.left=`${percent}%`;const track=box.querySelector("[role='meter']");track.setAttribute("aria-valuenow",String(percent));track.setAttribute("aria-valuetext",`${percent}分，${label}`);
}

function renderAgentTrace(result={}){
  const card=$("planner-agent-trace"),content=$("planner-agent-trace-content"),plan=result.agent_plan;
  if(!plan&&!result.tool_trace){card.hidden=true;return}
  card.hidden=false;
  const tools=(result.tool_trace||[]).map(item=>`<div class="agent-tool-row"><div><b>${escapePlannerText(item.tool||item.phase)}</b><small>${escapePlannerText(item.phase||"tool")}</small></div><strong>${escapePlannerText(item.status)}</strong></div>`).join("");
  content.innerHTML=`${plan?`<div class="agent-plan-summary"><div><span>结构化目标</span><strong>${escapePlannerText(plan.intent?.objective)}</strong></div><div><span>选定动作</span><strong>${escapePlannerText(plan.selected_action)}</strong></div><div><span>规划依据</span><strong>${escapePlannerText(plan.decision_summary)}</strong></div></div>`:""}<div class="agent-tool-list">${tools}</div>`;
}

function renderPlannerImprovement(point,overlay=null){
  const box=$("planner-improvement"),metrics=point.optimization_metrics||{},before=metrics.before||{},after=metrics.target_or_after||{};
  if(!Object.keys(before).length){box.hidden=true;return}
  box.hidden=false;$("planner-improvement-title").textContent=`点位 ${point.point_id} · ${point.optimization_class_label||"优化前后对比"}`;
  const regionTypes=new Set(["canopy_target","facility_review"]);
  const overlayRatio=overlay?(overlay.suggestion_regions||[]).filter(item=>regionTypes.has(item.type)).reduce((sum,item)=>sum+Number(item.pixel_ratio||0),0):null;
  const changeRatio=Math.min(.5,overlayRatio??Number(metrics.visual_change_ratio_target||0));
  $("planner-change-ratio").textContent=`${overlay?"GPU建议叠加区":"建议影响区目标"} ${(changeRatio*100).toFixed(1)}%`;
  const rows=[
    {label:"遮荫率 ↑",a:before.shade,b:after.shade,max:.7,d:1,pct:true},
    {label:"Tmrt ↓",a:before.tmrt,b:after.tmrt,max:75,d:1,unit:"°C"},
    {label:"UTCI ↓",a:before.utci,b:after.utci,max:55,d:1,unit:"°C"}
  ];
  $("planner-bar-chart").innerHTML=rows.map(row=>{const hasB=Number.isFinite(row.b),fmt=value=>row.pct?`${(value*100).toFixed(row.d)}%`:`${value.toFixed(row.d)}${row.unit||""}`;return `<div class="planner-chart-item"><div class="planner-chart-label"><span>${row.label}</span><span>现状 / ${hasB?"方案":"待量化"}</span></div><div class="planner-chart-track"><i class="planner-chart-before" style="width:${Math.min(100,row.a/row.max*100)}%"></i>${hasB?`<i class="planner-chart-after" style="width:${Math.min(100,row.b/row.max*100)}%"></i>`:""}</div><span class="planner-chart-value">${fmt(row.a)} → ${hasB?fmt(row.b):"—"}</span></div>`}).join("");
  $("planner-chart-note").textContent=overlay?"叠加区比例来自当前全景GPU语义定位；热指标为冻结标准化情景估计，并非施工后的实测值。":metrics.target_note||"选择点位后自动加载GPU规划叠加并更新影响区比例。";
}

async function loadPlannerCases() {
  const result = await api("/api/planner/cases");
  state.plannerCases = result.cases;
  const select = $("planner-case-select");
  result.cases.forEach(item => {
    const option = document.createElement("option");
    option.value = item.case_key || item.point_id;
    const status = item.pipeline === "planner_visual_perception_pilot" ? "高精度试点 · " : item.pipeline === "planner_visual_perception_terminal" ? `${item.terminal_label} · ` : "";
    option.textContent = `${status}点位 ${item.point_id} · ${item.priority} · ${item.recommended_action}`;
    select.appendChild(option);
  });
}

function showPlannerCase() {
  const selected = state.plannerCases.find(item => (item.case_key || item.point_id) === $("planner-case-select").value);
  if (!selected) return;
  const isPilot = selected.pipeline === "planner_visual_perception_pilot";
  const isTerminal = selected.pipeline === "planner_visual_perception_terminal";
  $("planner-result-title").textContent = `${isPilot ? "高精度试点 · " : isTerminal ? "终态评估 · " : ""}点位 ${selected.point_id} · ${selected.priority}`;
  const badge = $("planner-evidence-badge");
  badge.textContent = isPilot ? "视觉感知试点" : isTerminal ? selected.terminal_label : "正式审核案例"; badge.className = `evidence-badge formal ${isTerminal ? `terminal-${selected.terminal_tone}` : ""}`;
  const image = $("planner-main-image"); image.src = `${isPilot || isTerminal ? selected.card_url : (selected.phenology_review&&selected.panorama_url?selected.panorama_url:selected.card_url)}?v=${Date.now()}`; image.hidden = false;
  state.plannerOriginalImageUrl=image.src;showOverlayLegend();
  $("planner-image-empty").hidden = true;
  $("planner-metrics").innerHTML = selected.performance_available === false ? [
    metric("SVF_V2", selected.SVF_V2.toFixed(3)),
    metric("GVI_V2", selected.GVI_V2.toFixed(3)),
    metric("可见人行证据", `${(selected.retained_walkable_ratio_mean * 100).toFixed(1)}%`),
    metric("多视角树木证据", selected.season_aware_tree_evidence_count),
    metric("热改善量", selected.thermal_quantification?.display_value || "尚未量化")
  ].join("") : [
    metric("有效遮荫增加", `${(selected.mean_effective_shade_gain * 100).toFixed(1)}%`),
    metric("预计Tmrt变化", `${selected.estimated_mean_delta_tmrt_c.toFixed(2)}°C`),
    metric("预计UTCI变化", `${selected.estimated_mean_delta_utci_c.toFixed(2)}°C`),
    metric("优先级", selected.priority),
    metric("干预方式", selected.recommended_action)
  ].join("");
  const box = $("planner-recommendation"); box.hidden = false;
  const stageItem=isPilot||isTerminal?selected:(selected.phenology_review?{decision_stage:"stage_2_phenology_gate",decision_status:"暂停优化：人工复核确认现有树列或存在落叶期影响",optimization_eligible:false}:{decision_stage:"stage_4_optimization",decision_status:"正式审核案例：已进入优化与量化阶段",optimization_eligible:true});
  const visualLayer = isPilot && selected.visual_advice ? `<section class="planner-decision-layer visual-layer"><span>第一层 · 视觉规划建议</span><h3>${selected.visual_advice.headline}</h3><p>${selected.visual_advice.recommendation}</p><small>干预类型：${selected.visual_advice.intervention_type} · 不代表自动施工点</small></section>` : "";
  const thermalLayer = isPilot && selected.thermal_quantification ? `<section class="planner-decision-layer thermal-layer ${selected.thermal_quantification.status.startsWith("NOT_APPLICABLE") ? "not-applicable" : "withheld"}"><span>第二层 · 热效益量化</span><h3>${selected.thermal_quantification.display_value}</h3><p>${selected.thermal_quantification.reason}</p><small>${selected.thermal_quantification.status.startsWith("NOT_APPLICABLE") ? "当前无新增干预，因此不计算反事实收益" : "未通过冻结发布门槛，网页不显示内部试验数值"}</small></section>` : "";
  const performanceNote = selected.performance_available === false && !selected.thermal_quantification ? `<p><strong>量化边界：</strong>${selected.performance_unavailable_reason}</p>` : "";
  const sectorNote = isPilot && selected.audit_sector_centers_deg.length ? `<p><strong>复核扇区：</strong>${selected.audit_sector_centers_deg.map(value=>`${value}°`).join("、")}。${selected.claim_boundary}</p>` : isTerminal ? `<p><strong>终态边界：</strong>${selected.claim_boundary}</p>` : "";
  box.innerHTML = `${plannerDecisionPipeline(stageItem)}${visualLayer}${thermalLayer}${isPilot&&selected.visual_advice?"":`<h3>${selected.recommended_action}</h3><p>${selected.planner_recommendation}</p>`}<p><strong>证据说明：</strong>${selected.evidence_note}</p>${sectorNote}${performanceNote}`;
}

const OPTIMIZATION_META = {
  no_intervention:{label:"无需新增遮荫",color:"#25745a"},
  tree_growth:{label:"现有树木生长/养护",color:"#74a84a"},
  tree_priority:{label:"连续乔木优先",color:"#e08a2e"},
  canopy_gap:{label:"连接树冠缺口",color:"#d3a13f"},
  artificial_supplement:{label:"乔木优先·人工遮阳补充",color:"#8b67b2"},
  motor_only_review:{label:"机动车专用空间候选",color:"#66727a"}
};
const SHADE_META = {
  good:{label:"遮荫较好（SVF ≤ 0.15）",color:"#247a5a"},
  fair:{label:"遮荫一般（0.15–0.25）",color:"#73a84b"},
  poor:{label:"遮荫较差（0.25–0.35）",color:"#dda13b"},
  critical:{label:"遮荫不足（SVF > 0.35）",color:"#c95549"},
  seasonal:{label:"落叶/树种复核",color:"#71809c"}
};

function setPlannerMapLayer(layer){
  state.plannerMapLayer=layer==="shade"?"shade":"plan";
  ["plan","shade"].forEach(name=>{const button=$(`planner-map-layer-${name}`),active=name===state.plannerMapLayer;button.classList.toggle("active",active);button.setAttribute("aria-pressed",String(active))});
  updatePlannerMapLegend();drawPlannerMap();
}

function updatePlannerMapLegend(){
  const shade=state.plannerMapLayer==="shade",meta=shade?SHADE_META:OPTIMIZATION_META;
  $("planner-map-legend").innerHTML=Object.entries(meta).map(([key,item])=>`<span><i style="background:${item.color}"></i>${item.label}</span>`).join("");
  const result=state.plannerPointSummary;if(result)$("planner-map-title").textContent=`${result.point_count.toLocaleString()}点 · ${String(result.hour).padStart(2,"0")}:00`;
}

async function loadPlannerPoints() {
  if (state.plannerPointLoading) return;
  state.plannerPointLoading = true;
  $("planner-map-message").hidden = false; $("planner-map-message").textContent = "正在加载8,975个街景点…";
  try {
    const result = await api(`/api/planner/points?hour=${$("planner-map-hour").value}`);
    state.plannerPoints = result.points; state.plannerPointSummary = result; state.selectedPlannerPoint = null;
    resetPlannerMapView(false);
    updatePlannerMapLegend();
    const c = result.optimization_counts||{};
    $("planner-map-message").textContent = `无需新增 ${(c.no_intervention||0).toLocaleString()} · 树木生长 ${(c.tree_growth||0).toLocaleString()} · 乔木优先 ${(c.tree_priority||0).toLocaleString()} · 树冠补缺 ${(c.canopy_gap||0).toLocaleString()} · 人工遮阳辅助 ${(c.artificial_supplement||0).toLocaleString()} · 机动车空间复核 ${(c.motor_only_review||0).toLocaleString()}`;
    setTimeout(() => { $("planner-map-message").hidden = true; drawPlannerMap(); loadPlannerVisualLayers(); }, 900);
  } catch (error) { $("planner-map-message").textContent = error.message; }
  finally { state.plannerPointLoading = false; }
}

function plannerVisualRequest(explicitBbox=null){
  const context=[...state.context.bbox];
  if(explicitBbox)return{bbox:explicitBbox,rasterMaxDimension:1200,isOverview:true};
  const visible=state.plannerMapTransform?.viewExtent||context;
  const spanX=Math.max(1,visible[2]-visible[0]),spanY=Math.max(1,visible[3]-visible[1]);
  const contextSpan=Math.max(context[2]-context[0],context[3]-context[1]);
  if(Math.max(spanX,spanY)>=contextSpan*.78)return{bbox:context,rasterMaxDimension:1200,isOverview:true};
  const pad=Math.max(80,Math.max(spanX,spanY)*.16),grid=Math.max(20,Math.max(spanX,spanY)/12);
  const raw=[Math.max(context[0],visible[0]-pad),Math.max(context[1],visible[1]-pad),Math.min(context[2],visible[2]+pad),Math.min(context[3],visible[3]+pad)];
  const bbox=[Math.max(context[0],Math.floor(raw[0]/grid)*grid),Math.max(context[1],Math.floor(raw[1]/grid)*grid),Math.min(context[2],Math.ceil(raw[2]/grid)*grid),Math.min(context[3],Math.ceil(raw[3]/grid)*grid)];
  const rect=$("planner-point-map").getBoundingClientRect();
  const rasterMaxDimension=Math.min(1400,Math.max(900,Math.round(Math.max(rect.width,rect.height)*(window.devicePixelRatio||1)*1.1)));
  return{bbox,rasterMaxDimension,isOverview:false};
}

function schedulePlannerVisualRefresh(force=false){
  clearTimeout(state.plannerVisualTimer);
  state.plannerVisualTimer=setTimeout(()=>loadPlannerVisualLayers(force),360);
}

async function loadPlannerVisualLayers(force=false,explicitBbox=null){
  if(!state.context?.bbox)return;
  const {bbox,rasterMaxDimension,isOverview}=plannerVisualRequest(explicitBbox);
  const key=`building|${rasterMaxDimension}|${bbox.map(value=>Math.round(value)).join(",")}`;
  if(!force&&state.plannerVisualTile?.key===key)return;
  if(!force&&state.plannerVisualCache.has(key)){state.plannerVisualTile=state.plannerVisualCache.get(key);drawPlannerMap();return}
  const token=++state.plannerVisualToken,status=$("planner-layer-status");
  status.className="planner-layer-status loading";status.textContent=isOverview?"正在加载全域建筑概览…":"正在按当前视野提高清晰度…";
  try{
    const layers=await api("/api/visual-layers",{bbox,hour:14,max_buildings:100,raster_max_dimension:rasterMaxDimension,include_vectors:false});
    if(token!==state.plannerVisualToken)return;
    const image=await new Promise((resolve,reject)=>{const item=new Image();item.onload=()=>resolve(item);item.onerror=reject;item.src=`data:image/png;base64,${layers.building_png_base64}`});
    delete layers.building_png_base64;delete layers.shadow_png_base64;
    const tile={key,layers,buildingImage:image};state.plannerVisualTile=tile;if(isOverview)state.plannerOverviewVisualTile=tile;
    state.plannerVisualCache.set(key,tile);while(state.plannerVisualCache.size>8)state.plannerVisualCache.delete(state.plannerVisualCache.keys().next().value);
    const mpp=Math.max((bbox[2]-bbox[0])/layers.building_width,(bbox[3]-bbox[1])/layers.building_height);
    status.className="planner-layer-status ready";status.textContent=`${isOverview?"全域":"当前视野"}建筑已加载 · 约 ${mpp.toFixed(mpp<10?1:0)} 米/像素 · 缩放后自动更新`;
    drawPlannerMap();
  }catch(error){if(token!==state.plannerVisualToken)return;status.className="planner-layer-status error";status.textContent=`建筑图层加载失败：${error.message}`}
}

function drawPlannerMap() {
  const canvas = $("planner-point-map"); if (!canvas || !state.plannerPoints.length) return;
  const rect = canvas.getBoundingClientRect(); if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1; canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.fillStyle = "#edf0ea"; ctx.fillRect(0,0,rect.width,rect.height);
  const xs=state.plannerPoints.map(p=>p.x), ys=state.plannerPoints.map(p=>p.y);
  let minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const pad=Math.max(maxX-minX,maxY-minY)*.035; minX-=pad;maxX+=pad;minY-=pad;maxY+=pad;
  const baseS=Math.min((rect.width-24)/(maxX-minX),(rect.height-24)/(maxY-minY));
  const baseTx=(rect.width-(maxX-minX)*baseS)/2,baseTy=(rect.height-(maxY-minY)*baseS)/2;
  const zoom=state.plannerMapZoom,centerX=rect.width/2,centerY=rect.height/2;
  const project=p=>{
    const bx=baseTx+(p[0]-minX)*baseS,by=rect.height-(baseTy+(p[1]-minY)*baseS);
    return [centerX+(bx-centerX)*zoom+state.plannerMapPanX,centerY+(by-centerY)*zoom+state.plannerMapPanY];
  };
  const unproject=(sx,sy)=>{
    const bx=centerX+(sx-centerX-state.plannerMapPanX)/zoom,by=centerY+(sy-centerY-state.plannerMapPanY)/zoom;
    return[minX+(bx-baseTx)/baseS,minY+(rect.height-by-baseTy)/baseS];
  };
  const corners=[unproject(0,0),unproject(rect.width,0),unproject(0,rect.height),unproject(rect.width,rect.height)];
  const viewExtent=[Math.min(...corners.map(p=>p[0])),Math.min(...corners.map(p=>p[1])),Math.max(...corners.map(p=>p[0])),Math.max(...corners.map(p=>p[1]))];
  const buildingTile=tileCovers(state.plannerVisualTile,viewExtent)?state.plannerVisualTile:state.plannerOverviewVisualTile;
  if($("planner-show-buildings")?.checked&&buildingTile?.buildingImage&&buildingTile.layers?.building_extent){
    const extent=buildingTile.layers.building_extent,topLeft=project([extent[0],extent[3]]),bottomRight=project([extent[2],extent[1]]);
    ctx.drawImage(buildingTile.buildingImage,topLeft[0],topLeft[1],bottomRight[0]-topLeft[0],bottomRight[1]-topLeft[1]);
  }
  if(state.context){ctx.strokeStyle="rgba(132,143,137,.25)";ctx.lineWidth=.55;state.context.features.forEach(f=>{
    if(f.properties?.layer==="center_boundary")return;const lines=f.geometry.type==="LineString"?[f.geometry.coordinates]:f.geometry.coordinates;
    lines.forEach(line=>{ctx.beginPath();line.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke()});
  })}
  state.plannerPoints.forEach(point=>{const q=project([point.x,point.y]);if(q[0]<-8||q[0]>rect.width+8||q[1]<-8||q[1]>rect.height+8)return;ctx.beginPath();ctx.arc(q[0],q[1],point.formal_reviewed?4.2:Math.min(3.1,2.2+zoom*.18),0,Math.PI*2);const meta=state.plannerMapLayer==="shade"?(SHADE_META[point.grade]||SHADE_META.seasonal):(OPTIMIZATION_META[point.optimization_class]||OPTIMIZATION_META.motor_only_review);ctx.fillStyle=meta.color;ctx.globalAlpha=point.formal_reviewed?1:.78;ctx.fill();if(point.formal_reviewed){ctx.strokeStyle="#5c3b86";ctx.lineWidth=1.5;ctx.stroke()}});
  ctx.globalAlpha=1;
  if(state.selectedPlannerPoint){const p=state.selectedPlannerPoint,q=project([p.x,p.y]);ctx.beginPath();ctx.arc(q[0],q[1],8,0,Math.PI*2);ctx.strokeStyle="#172a24";ctx.lineWidth=2.5;ctx.stroke();}
  state.plannerMapTransform={project,unproject,minX,maxX,minY,maxY,width:rect.width,height:rect.height,baseS,zoom,viewExtent};
  $("planner-map-level").textContent=`${Math.round(zoom*100)}%`;
  updatePlannerScale(baseS*zoom);
}

function updatePlannerScale(pixelsPerMeter){
  if(!pixelsPerMeter)return;const target=90/pixelsPerMeter;const power=10**Math.floor(Math.log10(target));
  const value=[1,2,5,10].map(n=>n*power).reduce((a,b)=>Math.abs(b-target)<Math.abs(a-target)?b:a);
  const px=Math.max(35,Math.min(120,value*pixelsPerMeter));const box=$("planner-map-scale");box.querySelector("i").style.width=`${px}px`;
  box.querySelector("span").textContent=value>=1000?`${Number((value/1000).toFixed(value>=10000?0:1))} km`:`${Math.round(value)} m`;
}

function resetPlannerMapView(redraw=true){state.plannerMapZoom=1;state.plannerMapPanX=0;state.plannerMapPanY=0;if(redraw){drawPlannerMap();schedulePlannerVisualRefresh()}}

function zoomPlannerMap(factor,clientX=null,clientY=null){
  const canvas=$("planner-point-map"),rect=canvas.getBoundingClientRect();const old=state.plannerMapZoom;
  const next=Math.max(1,Math.min(18,old*factor));if(next===old)return;
  const sx=clientX===null?rect.width/2:clientX-rect.left,sy=clientY===null?rect.height/2:clientY-rect.top;
  const cx=rect.width/2,cy=rect.height/2,k=next/old;
  state.plannerMapPanX=(sx-cx)-(sx-cx-state.plannerMapPanX)*k;
  state.plannerMapPanY=(sy-cy)-(sy-cy-state.plannerMapPanY)*k;
  state.plannerMapZoom=next;drawPlannerMap();schedulePlannerVisualRefresh();
}

function beginPlannerMapDrag(event){
  if(event.button!==undefined&&event.button!==0)return;state.plannerMapPointer={id:event.pointerId,x:event.clientX,y:event.clientY,panX:state.plannerMapPanX,panY:state.plannerMapPanY};
  state.plannerMapDragged=false;try{event.currentTarget.setPointerCapture?.(event.pointerId)}catch{}event.currentTarget.classList.add("dragging");
}

function movePlannerMap(event){
  const drag=state.plannerMapPointer;if(!drag||drag.id!==event.pointerId)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
  if(Math.abs(dx)+Math.abs(dy)>4)state.plannerMapDragged=true;state.plannerMapPanX=drag.panX+dx;state.plannerMapPanY=drag.panY+dy;
  if(!state.plannerDrawFrame)state.plannerDrawFrame=requestAnimationFrame(()=>{state.plannerDrawFrame=null;drawPlannerMap()});
}

function endPlannerMapDrag(event){
  if(!state.plannerMapPointer||state.plannerMapPointer.id!==event.pointerId)return;try{event.currentTarget.releasePointerCapture?.(event.pointerId)}catch{}event.currentTarget.classList.remove("dragging");state.plannerMapPointer=null;schedulePlannerVisualRefresh();
}
function handlePlannerMapKey(event){
  const step=event.shiftKey?90:45;
  if(event.key==="ArrowLeft")state.plannerMapPanX+=step;
  else if(event.key==="ArrowRight")state.plannerMapPanX-=step;
  else if(event.key==="ArrowUp")state.plannerMapPanY+=step;
  else if(event.key==="ArrowDown")state.plannerMapPanY-=step;
  else if(event.key==="+"||event.key==="="){zoomPlannerMap(1.25);event.preventDefault();return}
  else if(event.key==="-"||event.key==="_"){zoomPlannerMap(1/1.25);event.preventDefault();return}
  else return;
  event.preventDefault();drawPlannerMap();schedulePlannerVisualRefresh();
}

function headingName(heading){
  return ({0:"北",30:"北偏东",60:"东北",90:"东",120:"东南",150:"南偏东",180:"南",210:"南偏西",240:"西南",270:"西",300:"西北",330:"北偏西"})[heading]||`${heading}°`;
}

function setPlannerViewMode(mode){
  state.plannerViewMode=mode;
  $("planner-direction-view").classList.toggle("active",mode==="direction");
  $("planner-fusion-view").classList.toggle("active",mode==="fusion");
  $("planner-direction-view").setAttribute("aria-selected",mode==="direction"?"true":"false");
  $("planner-fusion-view").setAttribute("aria-selected",mode==="fusion"?"true":"false");
  $("planner-direction-strip").hidden=mode!=="direction";
}

function loadPlannerImage(src){
  return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error("无法加载街景编辑图像"));image.src=src});
}

function renderPlannerMaskEditor(){
  const canvas=$("planner-mask-canvas"),ctx=canvas.getContext("2d"),base=state.plannerMaskBaseImage;
  if(!base||!state.plannerMaskLayer)return;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(base,0,0,canvas.width,canvas.height);
  ctx.save();ctx.globalAlpha=.38;ctx.drawImage(state.plannerMaskLayer,0,0);ctx.restore();
}

function plannerMaskSnapshot(){
  if(!state.plannerMaskLayer)return;const ctx=state.plannerMaskLayer.getContext("2d");
  state.plannerMaskHistory.push(ctx.getImageData(0,0,state.plannerMaskLayer.width,state.plannerMaskLayer.height));
  if(state.plannerMaskHistory.length>20)state.plannerMaskHistory.shift();
}

function setPlannerMaskMode(mode){
  state.plannerMaskMode=mode;
  for(const [id,value] of [["planner-brush-add","add"],["planner-brush-erase","erase"]]){
    const button=$(id);button.classList.toggle("active",mode===value);button.setAttribute("aria-pressed",mode===value?"true":"false");
  }
  $("planner-brush-cursor").classList.toggle("erase",mode==="erase");
}

function plannerMaskPoint(event){
  const canvas=$("planner-mask-canvas"),rect=canvas.getBoundingClientRect();
  return {
    x:Math.max(0,Math.min(canvas.width,(event.clientX-rect.left)*canvas.width/rect.width)),
    y:Math.max(0,Math.min(canvas.height,(event.clientY-rect.top)*canvas.height/rect.height))
  };
}

function updatePlannerBrushCursor(event){
  const canvas=$("planner-mask-canvas"),stage=canvas.parentElement,canvasRect=canvas.getBoundingClientRect(),stageRect=stage.getBoundingClientRect(),cursor=$("planner-brush-cursor");
  if(event.clientX<canvasRect.left||event.clientX>canvasRect.right||event.clientY<canvasRect.top||event.clientY>canvasRect.bottom){cursor.hidden=true;return}
  const displayDiameter=Number($("planner-brush-size").value)*canvasRect.width/canvas.width;
  cursor.style.width=`${Math.max(8,displayDiameter)}px`;cursor.style.height=`${Math.max(8,displayDiameter)}px`;
  cursor.style.left=`${event.clientX-stageRect.left}px`;cursor.style.top=`${event.clientY-stageRect.top}px`;cursor.hidden=false;
}

function paintPlannerMask(event){
  if(!state.plannerMaskDrawing||!state.plannerMaskLayer)return;
  const point=plannerMaskPoint(event),ctx=state.plannerMaskLayer.getContext("2d"),radius=Number($("planner-brush-size").value)/2;
  ctx.save();ctx.globalCompositeOperation=state.plannerMaskMode==="erase"?"destination-out":"source-over";
  ctx.fillStyle="rgba(31,166,94,1)";ctx.strokeStyle="rgba(31,166,94,1)";ctx.lineWidth=radius*2;ctx.lineCap="round";ctx.lineJoin="round";
  if(state.plannerMaskLastPoint){ctx.beginPath();ctx.moveTo(state.plannerMaskLastPoint.x,state.plannerMaskLastPoint.y);ctx.lineTo(point.x,point.y);ctx.stroke()}
  ctx.beginPath();ctx.arc(point.x,point.y,radius,0,Math.PI*2);ctx.fill();ctx.restore();state.plannerMaskLastPoint=point;
  state.plannerMaskDirty=true;state.plannerCommittedMask=null;$("planner-mask-submit").disabled=false;$("planner-editor-status").textContent="修改尚未提交";renderPlannerMaskEditor();
}

async function initializePlannerMaskEditor(){
  const result=state.plannerCurrentAnalysis;
  if(!result?.proposal_mask_png_base64||!state.plannerOriginalImageUrl)throw new Error("请先完成当前方位分析");
  const [base,maskImage]=await Promise.all([
    loadPlannerImage(state.plannerOriginalImageUrl),loadPlannerImage(`data:image/png;base64,${result.proposal_mask_png_base64}`)
  ]);
  const canvas=$("planner-mask-canvas");canvas.width=base.naturalWidth;canvas.height=base.naturalHeight;
  const maskLayer=document.createElement("canvas");maskLayer.width=canvas.width;maskLayer.height=canvas.height;
  const temp=document.createElement("canvas");temp.width=canvas.width;temp.height=canvas.height;
  const tempCtx=temp.getContext("2d",{willReadFrequently:true});tempCtx.drawImage(maskImage,0,0,canvas.width,canvas.height);
  const pixels=tempCtx.getImageData(0,0,canvas.width,canvas.height).data,maskCtx=maskLayer.getContext("2d");
  const overlay=maskCtx.createImageData(canvas.width,canvas.height);
  for(let i=0;i<pixels.length;i+=4){if(pixels[i]===1){overlay.data[i]=31;overlay.data[i+1]=166;overlay.data[i+2]=94;overlay.data[i+3]=255}}
  maskCtx.putImageData(overlay,0,0);state.plannerMaskBaseImage=base;state.plannerMaskLayer=maskLayer;
  state.plannerMaskAutomatic=maskCtx.getImageData(0,0,canvas.width,canvas.height);state.plannerMaskHistory=[];state.plannerMaskDirty=false;
  state.plannerCommittedMask=null;$("planner-mask-submit").disabled=true;$("planner-editor-status").textContent="使用自动区域";$("planner-mask-editor").hidden=false;setPlannerMaskMode("add");renderPlannerMaskEditor();
}

function plannerManualMaskPayload(){
  return state.plannerCommittedMask;
}

function submitPlannerMask(){
  if(!state.plannerMaskLayer||!state.plannerMaskDirty)return;
  const ctx=state.plannerMaskLayer.getContext("2d",{willReadFrequently:true}),pixels=ctx.getImageData(0,0,state.plannerMaskLayer.width,state.plannerMaskLayer.height).data;
  let active=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i]>0)active++;
  const ratio=active/(state.plannerMaskLayer.width*state.plannerMaskLayer.height);
  if(ratio<.003){$("planner-editor-status").textContent="区域过小，至少保留画面的0.3%";return}
  state.plannerCommittedMask=state.plannerMaskLayer.toDataURL("image/png").split(",",2)[1];
  state.plannerMaskDirty=false;$("planner-mask-submit").disabled=true;$("planner-editor-status").textContent=`修改已提交 · 覆盖画面${(ratio*100).toFixed(1)}%`;
  $("generate-realistic-tree").disabled=false;$("generate-realistic-tree").textContent="按已提交区域生成现实树木情景";
  updatePlannerFeedbackStatus();
}

function updatePlannerFeedbackStatus(){
  const checked=$("planner-feedback-consent").checked,status=$("planner-feedback-status");status.className="planner-feedback-status";
  if(!checked){status.textContent="未勾选：本次修改只用于当前生成，不会保存为训练反馈。";return}
  status.classList.add("armed");
  if($("planner-no-intervention").checked){status.textContent="已开启：提交“无需优化”结论和理由后，将写入本地RLHF/偏好学习候选集。";return}
  status.textContent=state.plannerCommittedMask?"已标记为反馈候选：生成并通过自动验收后，将写入本地RLHF/偏好学习数据集。":"默认开启：只有明确提交“无需优化”结论，或提交人工区域并生成验收通过后，才会真正保存。";
}

function resetPlannerMaskEditor(){
  $("planner-mask-editor").hidden=true;state.plannerMaskBaseImage=null;state.plannerMaskLayer=null;state.plannerMaskAutomatic=null;
  state.plannerMaskHistory=[];state.plannerMaskDirty=false;state.plannerCommittedMask=null;state.plannerMaskLastPoint=null;state.plannerCurrentAnalysis=null;state.plannerNoInterventionSubmitted=false;state.plannerNoInterventionReason="";
  $("planner-feedback-consent").checked=true;$("planner-feedback-panel").hidden=true;$("planner-review-decision").hidden=true;
  $("planner-no-intervention").checked=false;$("planner-no-intervention").disabled=false;$("planner-no-intervention-details").hidden=true;$("planner-no-intervention-reason").value="";$("planner-no-intervention-reason").disabled=false;
  const reviewSubmit=$("planner-no-intervention-submit");reviewSubmit.disabled=false;reviewSubmit.textContent="提交“无需优化”结论";
  const reviewStatus=$("planner-no-intervention-status");reviewStatus.className="planner-no-intervention-status";reviewStatus.textContent="";
  $("planner-feedback-proof").hidden=true;updatePlannerFeedbackStatus();
}

function showPlannerReviewControls(scope="direction"){
  state.plannerReviewScope=scope;
  const pointId=state.selectedPlannerPoint?.point_id;
  if(scope==="point"){
    $("planner-review-scope-help").textContent=`复核对象：点位 ${pointId||"—"} 的12方向融合全景与整体遮荫结论。`;
    $("planner-no-intervention-label").textContent="整个点位不需要遮荫优化";
    $("planner-no-intervention-reason").placeholder="例如：12个方向的现有连续树冠已覆盖主要慢行空间，整个点位无需新增遮荫设施。";
  }else{
    const heading=state.plannerDirectionHeading;
    $("planner-review-scope-help").textContent=`复核对象：点位 ${pointId||"—"} 的${headingName(Number(heading||0))} ${heading??0}°方位图。`;
    $("planner-no-intervention-label").textContent="当前方位不需要遮荫优化";
    $("planner-no-intervention-reason").placeholder="例如：现有连续高大乔木已完整覆盖该方位主要步行空间，方向天空缺口不构成热暴露风险。";
  }
  $("planner-review-decision").hidden=false;$("planner-feedback-panel").hidden=false;updatePlannerFeedbackStatus();
}

function setPlannerNoInterventionMode(){
  const active=$("planner-no-intervention").checked;
  $("planner-no-intervention-details").hidden=!active;
  const candidate=Boolean(state.plannerCurrentAnalysis?.generation_candidate_available??state.plannerCurrentAnalysis?.optimization_eligible);
  const hasAutomaticRegion=Boolean((state.plannerCurrentAnalysis?.suggestion_regions||[]).some(item=>item.type==="canopy_target"&&Number(item.pixel_ratio||0)>0));
  const hasTeacher=Boolean(state.plannerCurrentAnalysis?.teacher_reference_available);
  $("generate-realistic-tree").disabled=active||(!hasTeacher&&(!candidate||(!hasAutomaticRegion&&!state.plannerCommittedMask)));
  $("start-mask-edit").disabled=active||!candidate;
  updatePlannerFeedbackStatus();
  if(active)$("planner-no-intervention-reason").focus();
}

function showPlannerFeedbackProof(result, label){
  const status=$("planner-feedback-status"),proof=$("planner-feedback-proof");
  if(result.feedback_saved){
    status.className="planner-feedback-status saved";status.textContent=`已保存为RLHF/偏好学习候选 · ${result.feedback_id}`;
    proof.hidden=false;proof.innerHTML=`<b>反馈保存凭证</b><br>反馈ID：${escapePlannerText(result.feedback_id)}<br>类型：${escapePlannerText(label)}<br>训练状态：仅进入本地候选集，尚未执行在线训练。${result.feedback_verification_url?`<br><a href="${escapePlannerText(result.feedback_verification_url)}" target="_blank" rel="noopener">打开本地验证记录</a>`:""}`;
  }else{
    status.className="planner-feedback-status";status.textContent="结论已用于本次会话，但未写入RLHF/偏好学习候选集。";proof.hidden=true;
  }
}

async function submitPlannerNoIntervention(){
  const reason=$("planner-no-intervention-reason").value.trim();
  if(reason.length<5){$("planner-no-intervention-status").textContent="请至少填写5个字符，说明为什么不需要优化。";$("planner-no-intervention-reason").focus();return}
  if(state.plannerNoInterventionSubmitted&&reason===state.plannerNoInterventionReason){$("planner-no-intervention-status").textContent="该结论与理由已经提交；修改理由后可更新记录，或切换点位继续复核。";return}
  const button=$("planner-no-intervention-submit");button.disabled=true;button.textContent="正在提交并生成凭证…";
  try{
    let response;
    if(state.plannerSource==="upload"){
      const file=$("planner-upload").files[0];if(!file)throw new Error("上传图像已不可用，请重新选择图像");
      const form=new FormData();form.append("image",file);form.append("reason",reason);form.append("feedback_consent",$("planner-feedback-consent").checked?"1":"0");form.append("confidence",$("planner-confidence").value);
      response=await fetch("/api/planner/review-upload-no-intervention",{method:"POST",body:form,cache:"no-store"});
    }else{
      const point=state.selectedPlannerPoint,heading=state.plannerDirectionHeading;if(!point)throw new Error("请先选择点位");
      const reviewPath=state.plannerReviewScope==="point"?`/api/planner/points/${encodeURIComponent(point.point_id)}/review-no-intervention`:`/api/planner/points/${encodeURIComponent(point.point_id)}/directions/${heading}/review-no-intervention`;
      if(state.plannerReviewScope!=="point"&&heading===null)throw new Error("请先选择点位和方位图");
      response=await fetch(reviewPath,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reason,feedback_consent:$("planner-feedback-consent").checked,confidence:Number($("planner-confidence").value)}),cache:"no-store"});
    }
    const result=await response.json();if(!response.ok)throw new Error(result.error?.message||result.detail||"复核结论提交失败");
    state.plannerNoInterventionSubmitted=true;state.plannerNoInterventionReason=reason;$("planner-no-intervention-status").className="planner-no-intervention-status saved";$("planner-no-intervention-status").textContent=state.plannerReviewScope==="point"?"已确认：整个点位不需要遮荫优化。切换点位后可重新复核。":"已确认：当前方位不需要遮荫优化。切换点位或方位后可重新复核。";
    $("generate-realistic-tree").disabled=true;$("start-mask-edit").disabled=true;button.disabled=false;button.textContent="修改理由后可重新提交";showPlannerFeedbackProof(result,state.plannerReviewScope==="point"?"点位无需遮荫优化 + 自然语言理由":"方位无需遮荫优化 + 自然语言理由");
  }catch(error){$("planner-no-intervention-status").className="planner-no-intervention-status error";$("planner-no-intervention-status").textContent=error.message;button.disabled=false;button.textContent="重新提交“无需优化”结论"}
}

function resetPlannerOutputs(guidanceImage=null){
  state.plannerGuidanceImage=guidanceImage;state.plannerRealisticImage=null;state.plannerRealisticResult=null;
  if(guidanceImage){$("planner-main-image").src=guidanceImage;$("planner-main-image").hidden=false}
  const realistic=$("planner-realistic-image");realistic.hidden=true;realistic.removeAttribute("src");$("planner-realistic-empty").hidden=false;
  $("planner-model-alternatives").hidden=true;$("planner-model-alternative-buttons").innerHTML="";
}

function renderPlannerModelAlternatives(items=[],selectedModel=""){
  const panel=$("planner-model-alternatives"),buttons=$("planner-model-alternative-buttons");
  if(!items.length){panel.hidden=true;buttons.innerHTML="";return}
  panel.hidden=false;buttons.innerHTML=items.map((item,index)=>`<button type="button" class="${item.model===selectedModel?"active":""}" data-model-index="${index}"><b>${escapePlannerText(item.model)}</b><span>${item.quality_tier==="external_teacher_reference"?"教师参考":item.automatic_acceptance?"自动通过":"研究候选"}</span></button>`).join("");
  buttons.querySelectorAll("button").forEach(button=>button.onclick=()=>{const item=items[Number(button.dataset.modelIndex)];buttons.querySelectorAll("button").forEach(node=>node.classList.toggle("active",node===button));state.plannerRealisticImage=`data:image/png;base64,${item.image_png_base64}`;$("planner-realistic-image").src=state.plannerRealisticImage;$("planner-realistic-image").hidden=false;$("planner-realistic-empty").hidden=true});
}

function renderDirectionalResult(point,result){
  const heading=Number(result.heading??state.plannerDirectionHeading),r=result.ratios||{},geometry=result.road_geometry_evidence||{},depth=result.depth_cross_section_evidence||{},osm=result.osm_road_context||{},reference=result.self_supervised_reference||{},thermal=result.thermal_reference_estimate||{};
  state.plannerViewMode="direction";setPlannerViewMode("direction");
  resetPlannerMaskEditor();state.plannerCurrentAnalysis=result;state.plannerOriginalImageUrl=`/api/planner/points/${encodeURIComponent(point.point_id)}/directions/${heading}/image`;
  const guidance=`data:image/png;base64,${result.overlay_png_base64}`;resetPlannerOutputs(guidance);$("planner-main-image").src=guidance;
  const candidate=Boolean(result.generation_candidate_available??result.optimization_eligible),hasAutomaticRegion=Boolean((result.suggestion_regions||[]).some(item=>item.type==="canopy_target"&&Number(item.pixel_ratio||0)>0));
  const hasTeacher=Boolean(result.teacher_reference_available);
  const realisticButton=$("generate-realistic-tree");realisticButton.hidden=false;realisticButton.disabled=!candidate||!hasAutomaticRegion;
  realisticButton.disabled=!hasTeacher&&(!candidate||!hasAutomaticRegion);
  realisticButton.textContent=hasTeacher?"查看Image2教师参考":!candidate?"当前方位不建议新增遮荫":!hasAutomaticRegion?"请先微调树冠区域":"生成当前方位现实树木情景";
  const editButton=$("start-mask-edit");editButton.hidden=!candidate;editButton.disabled=!candidate;
  showPlannerReviewControls();
  showOverlayLegend(result.overlay_legend);renderPublicSpaceMeter(result.public_space_score,result.space_type_assessment);
  $("planner-result-title").textContent=`点位 ${point.point_id} · ${headingName(heading)} ${heading}°方位`;
  $("planner-evidence-badge").textContent="方位图GPU判断";$("planner-evidence-badge").className="evidence-badge formal";
  $("planner-direction-note").textContent=`${headingName(heading)} ${heading}°：道路尽头已排除；仅在画面左右两侧定位方案。`;
  $("planner-metrics").innerHTML=[
    metric("预计遮荫提升",Number.isFinite(thermal.estimated_shade_gain)?`+${(thermal.estimated_shade_gain*100).toFixed(1)}%`:"方案后量化"),
    metric("预计Tmrt变化",Number.isFinite(thermal.estimated_delta_tmrt_c)?`${thermal.estimated_delta_tmrt_c.toFixed(2)}°C`:"待量化"),
    metric("预计UTCI变化",Number.isFinite(thermal.estimated_delta_utci_c)?`${thermal.estimated_delta_utci_c.toFixed(2)}°C`:"待量化")
  ].join("");
  const weather=thermal.scenario_weather||{};
  const refs=(reference.good_shade_references||[]).slice(0,3).map(item=>`点${item.point_id}（${item.reference_azimuth_deg}°，相似度${item.cosine_similarity}）`).join("、");
  const box=$("planner-recommendation");box.hidden=false;box.innerHTML=`${plannerDecisionPipeline(result)}<h3>${escapePlannerText(result.planning_priority)}</h3><p>${escapePlannerText(result.recommendation)}</p><div class="planner-brief-facts"><span>天空视域 <b>${Number(result.svf).toFixed(3)}</b></span><span>道路尽头保护 <b>${(Number(geometry.road_end_protected_ratio||0)*100).toFixed(1)}%</b></span><span>侧向连续性 <b>${Number(depth.directional_side_shade_continuity||0).toFixed(2)}</b></span></div><details><summary>查看多模态判断依据</summary><p><strong>OSM道路先验：</strong>${escapePlannerText(osm.fclass||"unknown")}${osm.road_name?` · ${escapePlannerText(osm.road_name)}`:""}；路沿透视${Number(geometry.curb_perspective_strength||0).toFixed(2)}，车道线置信${Number(geometry.lane_marking_confidence||0).toFixed(2)}。</p><p><strong>道路/采样车宽比（道路/采样车头宽度比）：</strong>${Number(geometry.road_to_ego_hood_width_ratio||0).toFixed(2)}；<strong>道路两侧连续性：</strong>${Number(depth.directional_side_shade_continuity||0).toFixed(2)}；慢行带${(Number(depth.walk_corridor_ratio||0)*100).toFixed(2)}%，${result.dynamic_active_mobility_detected?"检测到行人/两轮目标。":"未检测到动态目标，但不据此否定慢行需求。"}</p><p><strong>参考：</strong>${escapePlannerText(reference.learned_shade_advantage||"DINOv2相似案例检索")} ${refs?escapePlannerText(refs):""}。</p><p><strong>气象：</strong>${weather.scenario_date||"2024-08-04"} ${weather.hour??14}:00南京晴热情景；结果是形态迁移估计，不是实测。</p></details>`;
  renderPlannerImprovement(point,result);
}

async function analyzePlannerDirection(point,heading){
  setPlannerViewMode("direction");state.plannerDirectionHeading=Number(heading);
  document.querySelectorAll(".planner-direction-button").forEach(button=>button.classList.toggle("active",Number(button.dataset.heading)===Number(heading)));
  const cacheKey=`${heading}:${$("planner-confidence").value}:${$("planner-map-hour").value}`;
  if(state.plannerDirectionResults.has(cacheKey)){renderDirectionalResult(point,state.plannerDirectionResults.get(cacheKey));return}
  $("planner-evidence-badge").textContent="方位图GPU分析中";$("planner-direction-note").textContent=`正在独立分析${headingName(Number(heading))} ${heading}°方位…`;
  beginPlannerProgress(`${headingName(Number(heading))} ${heading}°方位分析`,["加载方位图与道路先验","GPU语义分割与深度估计","检索相似优质遮荫案例","生成并检查规划区域"],28);
  try{
    const response=await fetch(`/api/planner/points/${encodeURIComponent(point.point_id)}/directions/${heading}/analyze?confidence=${$("planner-confidence").value}&hour=${$("planner-map-hour").value}`,{method:"POST",cache:"no-store"});
    const result=await response.json();if(!response.ok)throw new Error(result.error?.message||"方位图分析失败");
    state.plannerDirectionResults.set(cacheKey,result);renderDirectionalResult(point,result);finishPlannerProgress(`${headingName(Number(heading))} ${heading}°方位分析完成`);
  }catch(error){failPlannerProgress(error.message);throw error}
}

async function loadPlannerPointDirections(point){
  const panel=$("planner-direction-panel");panel.hidden=false;state.plannerDirectionResults=new Map();state.plannerDirectionBatch=null;
  const manifest=await api(`/api/planner/points/${encodeURIComponent(point.point_id)}/directions`);state.plannerDirections=manifest.directions;
  const teacherHeadings=new Set((point.teacher_reference_headings||[]).map(Number));
  $("planner-direction-strip").innerHTML=manifest.directions.map(item=>`<button class="planner-direction-button ${teacherHeadings.has(Number(item.heading))?"teacher-ready":""}" data-heading="${item.heading}" title="${teacherHeadings.has(Number(item.heading))?"可查看Image2教师参考 · ":""}分析${headingName(item.heading)}方向"><b>${headingName(item.heading)}</b><span>${item.heading}°${teacherHeadings.has(Number(item.heading))?" · Image2":""}</span></button>`).join("");
  document.querySelectorAll(".planner-direction-button").forEach(button=>button.onclick=()=>analyzePlannerDirection(point,Number(button.dataset.heading)).catch(error=>{$("planner-direction-note").textContent=error.message}));
  await analyzePlannerDirection(point,(point.teacher_reference_headings||[])[0] ?? 0);
}

async function loadPlannerDirectionFusion(point){
  setPlannerViewMode("fusion");$("planner-evidence-badge").textContent="12方向融合中";$("planner-direction-note").textContent="正在依次分析12张原始方位图，并将方案掩膜投影到全景总览…";
  beginPlannerProgress("12方向独立分析与全景融合",["准备12张原始方位图","逐方向运行语义、深度和道路判断","汇总方向遮荫缺口","按已知heading融合规划全景","执行接缝与结果检查"],110);
  try{if(!state.plannerDirectionBatch){
    const response=await fetch(`/api/planner/points/${encodeURIComponent(point.point_id)}/directions/analyze-all?confidence=${$("planner-confidence").value}&hour=${$("planner-map-hour").value}`,{method:"POST",cache:"no-store"});
    const result=await response.json();if(!response.ok)throw new Error(result.error?.message||"12方向融合失败");state.plannerDirectionBatch=result;
  }}catch(error){failPlannerProgress(error.message);throw error}
  const result=state.plannerDirectionBatch;resetPlannerMaskEditor();const guidance=`data:image/png;base64,${result.fused_panorama_png_base64}`;resetPlannerOutputs(guidance);$("planner-main-image").src=guidance;$("generate-realistic-tree").hidden=true;$("start-mask-edit").hidden=true;
  showOverlayLegend({canopy_target:"树冠目标覆盖区域",facility_review:"人工遮阳棚候选覆盖区域"});
  $("planner-result-title").textContent=`点位 ${point.point_id} · 12方向决策融合全景`;
  $("planner-evidence-badge").textContent="12方向融合完成";$("planner-evidence-badge").className="evidence-badge formal";
  const counts=result.direction_need_counts||{};$("planner-direction-note").textContent=`12个方向均独立判断：通常无需优化${counts["通常不需要"]||0}个、较低${counts["较低"]||0}个、中等${counts["中等"]||0}个、较高${counts["较高"]||0}个、季节复核${counts["季节复核"]||0}个。`;
  const continuity=result.side_shade_continuity||{};
  $("planner-metrics").innerHTML=[metric("点位全景SVF",Number(result.point_level_svf).toFixed(3)),metric("独立方位数",result.direction_count),metric("侧向遮荫连续性",Number(continuity.mean||0).toFixed(2)),metric("局部缺口方向",continuity.local_gap_direction_count||0),metric("良好参考门槛","SVF < 0.30"),metric("方案依据","12方位独立融合")].join("");
  const box=$("planner-recommendation");box.hidden=false;box.innerHTML=`<h3>全景仅用于汇总展示</h3><p>绿色或橙色方案区域均来自12张原始方位图的独立路沿透视、道路尽头排除、左右侧判断和已有设施参照。原始方位图RGB与方案mask采用同一heading投影、中心优先羽化和接缝平滑重新融合。</p><p><strong>道路两侧连续性：</strong>${escapePlannerText(continuity.assessment||"等待统计")}；局部缺口方位：${(continuity.local_gap_headings||[]).length?(continuity.local_gap_headings||[]).join("°、")+"°":"无"}。即使点位平均SVF较低，局部方位缺口仍会单独保留。</p><p><strong>方向统计：</strong>${escapePlannerText($("planner-direction-note").textContent)}</p>`;
  showPlannerReviewControls("point");finishPlannerProgress("12个方向已完成独立分析，融合全景与点位级复核已就绪");
}

async function loadPlannerPointOverlay(point){
  const token=++state.plannerOverlayToken,badge=$("planner-evidence-badge");badge.textContent="GPU规划定位中";
  try{
    const confidence=Number($("planner-confidence").value);const response=await fetch(`/api/planner/points/${encodeURIComponent(point.point_id)}/overlay?confidence=${confidence}`,{method:"POST",cache:"no-store"});
    const result=await response.json();if(!response.ok)throw new Error(result.error?.message||"规划叠加生成失败");
    if(token!==state.plannerOverlayToken||state.selectedPlannerPoint?.point_id!==point.point_id)return;
    $("planner-main-image").src=`data:image/png;base64,${result.overlay_png_base64}`;showOverlayLegend(result.overlay_legend);renderPlannerImprovement(point,result);renderPublicSpaceMeter(result.public_space_score,result.space_type_assessment);
    badge.textContent="GPU证据叠加完成";badge.className="evidence-badge formal";
    const box=$("planner-recommendation");box.insertAdjacentHTML("beforeend",`<p><strong>空间类型：</strong>${escapePlannerText(result.space_type_assessment)}（概率性判断；公共步行骑行可能性：${escapePlannerText(result.public_walk_cycle_likelihood)}）</p>`);
  }catch(error){if(token!==state.plannerOverlayToken)return;badge.textContent="基础评估可用";$("planner-chart-note").textContent=`GPU叠加暂未完成：${error.message}；基础指标与规划等级仍可使用。`;}
}

function selectPlannerPoint(point) {
  state.selectedPlannerPoint = point; drawPlannerMap();
  $("planner-result-title").textContent = `点位 ${point.point_id} · ${point.grade_label}`;
  const badge=$("planner-evidence-badge"); badge.textContent=point.formal_reviewed?"正式审核点":"全样本自动初筛";badge.className=`evidence-badge ${point.formal_reviewed?"formal":"preliminary"}`;
  resetPlannerMaskEditor();const image=$("planner-main-image");image.src=point.panorama_url;image.hidden=false;$("planner-image-empty").hidden=true;resetPlannerOutputs(point.panorama_url);
  state.plannerOriginalImageUrl=point.panorama_url;showOverlayLegend();
  renderPublicSpaceMeter(point.public_space_score,point.space_type_assessment||"概率判断");
  const perf=point.performance_estimate||{},numeric=Number.isFinite(perf.effective_shade_gain);
  $("planner-metrics").innerHTML=[metric("预计遮荫提升",numeric?`+${(perf.effective_shade_gain*100).toFixed(1)}%`:"待量化"),metric("预计Tmrt变化",Number.isFinite(perf.estimated_delta_tmrt_c)?`${perf.estimated_delta_tmrt_c.toFixed(2)}°C`:"待量化"),metric("预计UTCI变化",Number.isFinite(perf.estimated_delta_utci_c)?`${perf.estimated_delta_utci_c.toFixed(2)}°C`:"待量化")].join("");
  const teacherNote=(point.teacher_reference_headings||[]).length?`<p><strong>Image2教师参考：</strong>该点有 ${(point.teacher_reference_headings||[]).length} 个方位已生成教师样本，系统会优先打开 ${point.teacher_reference_headings[0]}° 方位。</p>`:"";
  const box=$("planner-recommendation");box.hidden=false;box.innerHTML=`${plannerDecisionPipeline(point)}<h3>${escapePlannerText(point.optimization_class_label||point.grade_label)}</h3><p>${escapePlannerText(point.recommended_action)}</p>${teacherNote}<div class="planner-brief-facts"><span>SVF <b>${point.svf.toFixed(3)}</b></span><span>空间 <b>${escapePlannerText(point.space_type_assessment||"概率判断中")}</b></span><span>慢行 <b>${escapePlannerText(point.public_walk_cycle_likelihood||"可能")}</b></span></div><details><summary>查看判断依据与适用边界</summary><p>${escapePlannerText(point.planner_citywide_basis||"SVF、物候、语义、深度与多视角树木证据。")} 树木方向与共现指标是多视角代理证据，不等同于实际树冠投影。</p></details>`;
  const planButton=$("generate-point-plan");planButton.hidden=false;planButton.textContent="分析12方向并融合全景总览";
  $("generate-realistic-tree").hidden=true;
  $("start-mask-edit").hidden=true;
  renderPlannerImprovement(point);renderAgentTrace({});loadPlannerPointDirections(point).catch(error=>{$("planner-direction-note").textContent=error.message});
}

function handlePlannerMapClick(event) {
  if(state.plannerMapDragged){state.plannerMapDragged=false;return}if(!state.plannerMapTransform)return;const rect=$("planner-point-map").getBoundingClientRect();const sx=event.clientX-rect.left,sy=event.clientY-rect.top;
  let best=null,bestDistance=Infinity;state.plannerPoints.forEach(point=>{const q=state.plannerMapTransform.project([point.x,point.y]);const d=(q[0]-sx)**2+(q[1]-sy)**2;if(d<bestDistance){bestDistance=d;best=point}});
  if(best && bestDistance<=225)selectPlannerPoint(best);else $("planner-map-message").hidden=false,$("planner-map-message").textContent="请点击更靠近彩色点位的位置";
}

async function generatePointPlan() {
  const p=state.selectedPlannerPoint;if(!p)return;
  await loadPlannerDirectionFusion(p);return;
  const formal=p.formal_case;
  const perf=p.performance_estimate||{};
  let benefit;
  if(formal){benefit=`正式三维树冠情景估计：有效遮荫增加 ${(formal.mean_effective_shade_gain*100).toFixed(1)}%，Tmrt ${formal.estimated_mean_delta_tmrt_c.toFixed(2)}°C，UTCI ${formal.estimated_mean_delta_utci_c.toFixed(2)}°C。`}
  else if(perf.status==="estimated_standardized_intervention"){benefit=`标准化规划情景估计：Shade +${(perf.effective_shade_gain*100).toFixed(1)}%，Tmrt ${perf.estimated_delta_tmrt_c.toFixed(2)}±${perf.delta_tmrt_response_seed_std_c.toFixed(2)}°C，UTCI ${perf.estimated_delta_utci_c.toFixed(2)}±${perf.delta_utci_response_seed_std_c.toFixed(2)}°C。该值表示提升至本小时全市遮荫上四分位基准、且单小时增量不超过20%的模型估计。`}
  else if(perf.status==="no_intervention_needed"){benefit="当前遮荫已达到本小时全市上四分位基准，标准化情景不触发新增干预：Shade、Tmrt和UTCI预计变化均为0。"}
  else{benefit="该点已完成遮荫现状与概率性空间类型判断；热响应尚未通过冻结量化链，因此保留为待量化，不输出虚假收益数值。"}
  const plan=p.grade==="good"?"保持现有遮荫结构，检查树冠连续性、季节变化与人行道实际覆盖，原则上不新增设施。":p.recommended_action;
  const box=$("planner-recommendation"),button=$("generate-point-plan");box.hidden=false;box.innerHTML=`${plannerDecisionPipeline(p)}<h3>点位 ${p.point_id} 遮荫优化方案</h3><p><strong>1. 遮荫评估：</strong>SVF ${p.svf.toFixed(3)}；${$("planner-map-hour").value}:00遮荫率 ${(p.shade*100).toFixed(1)}%作为辅助，GVI ${p.gvi.toFixed(3)}。</p><p><strong>2. 空间类型：</strong>${p.space_type_assessment||"城市道路空间"}，公共步行骑行可能性${p.public_walk_cycle_likelihood||"可能"}；该判断位于遮荫评估之后。</p><p><strong>3. 树木优先方案：</strong>${plan}</p><p><strong>4. 人工遮阳：</strong>仅在2.5D布局确认不适合连续乔木时，转为结合建筑外侧或既有道路设施的补充复核。</p><p><strong>5. 性能说明：</strong>${benefit}</p><p><strong>工具执行：</strong>Agent正在根据自然语言约束选择GPU语义、视觉语言、深度布局、生成和验收工具。</p>`;
  button.disabled=true;button.textContent="GPU定位优化区域中…";
  try{
    const confidence=Number($("planner-confidence").value);const response=await fetch(`/api/planner/points/${encodeURIComponent(p.point_id)}/analyze?confidence=${confidence}`,{method:"POST",cache:"no-store"});
    const result=await response.json();if(!response.ok)throw new Error(result.error?.message||"建议区域生成失败");
    $("planner-main-image").src=result.generation_available?`data:image/png;base64,${result.generated_panorama_png_base64}`:`data:image/png;base64,${result.overlay_png_base64}`;showOverlayLegend(result.generation_available?{}:result.overlay_legend);
    renderAgentTrace(result);renderPlannerImprovement(p,result);
    const regionText=(result.suggestion_regions||[]).map(region=>`${region.label} ${(region.pixel_ratio*100).toFixed(1)}%`).join("、");
    const agentText=result.generation_available?`多模态Agent已生成并自动验收全景情景；SVF预计减少 ${result.scenario_metrics.svf_reduction.toFixed(3)}，植被占比增加 ${(result.scenario_metrics.vegetation_ratio_increase*100).toFixed(1)}%。Tmrt/UTCI仅在冻结多日期特征链重算后报告。`:`多模态Agent状态：${result.agent_state}。${result.agent_explanation||"未通过正式生成门槛。"}`;
    box.innerHTML+=`<p><strong>Agent结果：</strong>${agentText}</p>${result.generation_available?"":`<p><strong>半透明建议层：</strong>${regionText||"当前没有足够可靠的自动候选区域"}。</p>`}`;
  }catch(error){box.innerHTML+=`<p><strong>图像定位失败：</strong>${error.message}</p>`}
  finally{button.disabled=false;button.textContent="重新生成该点遮荫优化方案"}
}

async function generateRealisticTreeScenario(){
  if(!$("planner-mask-editor").hidden&&state.plannerMaskDirty&&!state.plannerCommittedMask){
    $("planner-editor-status").textContent="请先点击“提交本次修改”，再生成情景";$("planner-mask-submit").focus();return;
  }
  const button=$("generate-realistic-tree");button.disabled=true;button.textContent="全GPU生成并自动验收中…";
  $("planner-evidence-badge").textContent="现实情景生成中";
  beginPlannerProgress("多模型树冠情景生成",["准备语义、道路尽头和人工区域条件","PowerPaint生成候选","Paint-by-Example生成候选","重新语义分割并检查建筑伪影","比较候选并决定是否发布"],170);
  try{
    let result;
    if(state.plannerSource==="upload"){
      result=await analyzePlannerUpload(true);if(!result)return;
    }else{
      const point=state.selectedPlannerPoint,heading=state.plannerDirectionHeading;if(!point||heading===null)throw new Error("请先选择点位和一个方位");
      const manualMask=plannerManualMaskPayload(),feedbackConsent=$("planner-feedback-consent").checked;
      const path=manualMask?`/api/planner/points/${encodeURIComponent(point.point_id)}/directions/${heading}/generate-realistic-edited`:`/api/planner/points/${encodeURIComponent(point.point_id)}/directions/${heading}/generate-realistic?confidence=${$("planner-confidence").value}&hour=${$("planner-map-hour").value}`;
      const options={method:"POST",cache:"no-store"};
      if(manualMask){options.headers={"Content-Type":"application/json"};options.body=JSON.stringify({confidence:Number($("planner-confidence").value),hour:Number($("planner-map-hour").value),manual_mask_png_base64:manualMask,feedback_consent:feedbackConsent})}
      const response=await fetch(path,options);
      result=await response.json();if(!response.ok)throw new Error(result.error?.message||result.detail||"现实树木情景生成失败");
    }
    state.plannerRealisticResult=result;
    renderPlannerModelAlternatives(result.model_alternatives||[],result.generator||"");
    if(!result.generation_available){
      const rows=result.model_comparison||[];const diagnostic=[...new Set(rows.map(item=>item.generator_model))].map(model=>{const group=rows.filter(item=>item.generator_model===model),best=group.slice().sort((a,b)=>Number(b.score||-99)-Number(a.score||-99))[0]||{};const failed=[!best.target_region_fill_pass?"未填满区域":"",!best.tree_semantic_pass?"树冠语义不足":"",!best.building_hallucination_pass?"建筑/广告牌误生成":""].filter(Boolean).join("、")||"综合分不足";return `${escapePlannerText(model)}：${failed}`}).join("；");if(diagnostic)$("planner-recommendation").insertAdjacentHTML("beforeend",`<details open><summary>多模型未通过原因</summary><p>${diagnostic}。低质量候选已被阻止发布。</p></details>`);throw new Error(result.agent_explanation||"生成候选未通过自动验收");
    }
    state.plannerRealisticImage=`data:image/png;base64,${result.realistic_tree_png_base64}`;
    $("planner-realistic-image").src=state.plannerRealisticImage;$("planner-realistic-image").hidden=false;$("planner-realistic-empty").hidden=true;
    const reviewRequired=Boolean(result.planner_review_required||!result.automatic_acceptance);
    $("planner-evidence-badge").textContent=reviewRequired?"现实情景待规划师复核":"现实情景已自动验收";$("planner-evidence-badge").className=`evidence-badge ${reviewRequired?"preliminary":"formal"}`;
    const reference=result.reference_tree_case||{},qc=result.scenario_qc||{};
    const materialMethod={same_image_most_obvious_crown_first:"本张图片中最明显的语义树冠",same_image_semantic_most_obvious_crown:"本张图片中最明显的语义树冠",nearby_coordinate_most_obvious_crown_fallback:"周边坐标优质街景中最明显的树冠",nearest_coordinate_semantic_crown_fallback:"距当前坐标最近的优质语义树冠",same_image_weak_crown_last_resort:"本张图片中的弱树冠证据"}[result.material_source_method]||"本张图片优先、周边坐标加权补充的真实树冠素材";
    const comparisons=result.model_comparison||[];const comparisonText=[...new Set(comparisons.map(item=>item.generator_model))].map(model=>{const rows=comparisons.filter(item=>item.generator_model===model),passed=rows.filter(item=>item.automatic_acceptance),review=rows.filter(item=>item.planner_review_candidate&&!item.automatic_acceptance),best=rows.slice().sort((a,b)=>Number(b.score||-99)-Number(a.score||-99))[0]||{};return `${escapePlannerText(model)}：自动通过${passed.length}、待复核${review.length}，最佳有效编辑${(Number(best.changed_fraction_inside_mask||0)*100).toFixed(0)}%`}).join("；");
    $("planner-recommendation").insertAdjacentHTML("beforeend",`<p><strong>现实树木情景${reviewRequired?"（待复核候选）":"（自动通过）"}：</strong>使用${escapePlannerText(result.generator||"本地扩散模型")}直接修补原始街景，仅编辑${result.manual_mask_used?"规划师微调后的":"自动定位的"}树冠区域；不向原图预先复制或粘贴树冠。参考来源为${escapePlannerText(materialMethod)}${reference.geographic_distance_m!=null?`（备用案例距当前点约${Number(reference.geographic_distance_m).toFixed(0)}m）`:""}。目标区域有效编辑${(Number(qc.changed_fraction_inside_mask||0)*100).toFixed(0)}%，目标区树木语义占比由${(Number(qc.target_vegetation_fraction_before||0)*100).toFixed(0)}%变为${(Number(qc.target_vegetation_fraction_after||0)*100).toFixed(0)}%，全图植被变化${Number(qc.vegetation_gain||0)>=0?"+":""}${(Number(qc.vegetation_gain||0)*100).toFixed(1)}%，建筑变化${Number(qc.building_gain||0)>=0?"+":""}${(Number(qc.building_gain||0)*100).toFixed(1)}%，掩膜外最大变化${qc.outside_mask_max_difference??0}。${reviewRequired?"该图仅供规划师比较和修正，不作为自动验收结论。":"该图已通过自动质量门槛。"}${result.feedback_saved?` 人工反馈已在本地保存（${escapePlannerText(result.feedback_id)}），供后续偏好学习/RLHF使用。`:" 本次结果未写入反馈训练集。"}</p>${comparisonText?`<details><summary>查看生成质量分级</summary><p>${comparisonText}。自动验收依据独立语义分割复核，而不是易受季节和光照影响的绿色像素阈值。</p></details>`:""}`);
    const feedbackStatus=$("planner-feedback-status"),proof=$("planner-feedback-proof");
    if(result.feedback_saved){feedbackStatus.className="planner-feedback-status saved";feedbackStatus.textContent=`已保存为RLHF/偏好学习候选 · ${result.feedback_id}`;proof.hidden=false;proof.innerHTML=`<b>反馈保存凭证</b><br>反馈ID：${escapePlannerText(result.feedback_id)}<br>内容：原图、自动区域、人工提交区域、候选情景与质量分级<br>训练状态：仅进入本地候选集，尚未执行在线训练。${result.feedback_verification_url?`<br><a href="${escapePlannerText(result.feedback_verification_url)}" target="_blank" rel="noopener">打开本地验证记录</a>`:""}`}
    else if($("planner-feedback-consent").checked){feedbackStatus.className="planner-feedback-status armed";feedbackStatus.textContent="未保存：只有已提交的人工修改且生成结果通过验收后，才会形成反馈样本。"}
    finishPlannerProgress(reviewRequired?"多模型比较完成，已显示待规划师复核候选":"多模型候选比较与自动验收完成，合格情景已发布");
  }catch(error){
    failPlannerProgress(`未发布：${error.message}`);
    $("planner-evidence-badge").textContent="保留规划指导图";
    const box=$("planner-recommendation");box.hidden=false;box.insertAdjacentHTML("beforeend",`<p><strong>现实情景未发布：</strong>${escapePlannerText(error.message)}。规划指导图仍是正式可审计结果。</p>`);
    if($("planner-feedback-consent").checked){const status=$("planner-feedback-status");status.className="planner-feedback-status armed";status.textContent="尚未保存反馈：本次现实情景未通过验收，请调整区域后重新提交生成。"}
  }finally{
    button.disabled=false;button.textContent=state.plannerCommittedMask?"按已提交区域重新生成现实树木情景":"重新生成当前方位现实树木情景";
  }
}

async function analyzePlannerUpload(generateRealistic=false) {
  const input = $("planner-upload"), button = $("analyze-planner-upload");
  if (!input.files.length) return alert("请先选择一张街景图像。");
  const file = input.files[0];
  const preview = $("planner-main-image"); preview.src = URL.createObjectURL(file); preview.hidden = false;
  state.plannerOriginalImageUrl=preview.src;showOverlayLegend();
  $("planner-direction-panel").hidden=true;
  $("planner-image-empty").hidden = true;
  $("planner-result-title").textContent = "正在运行本地多模态规划Agent…";
  $("planner-evidence-badge").textContent = "处理中";
  button.disabled = true; button.textContent = generateRealistic?"现实树木情景生成中…":"多模态Agent分析与自动验收中…";
  if(!generateRealistic)beginPlannerProgress("上传街景多模态分析",["读取并校验方位图","GPU语义分割与深度估计","检索南京相似街景","规划遮荫候选并自动检查"],45);
  try {
    const manualMask=generateRealistic?plannerManualMaskPayload():null;
    const form = new FormData(); form.append("image", file); form.append("planner_intent", $("planner-intent").value.trim());form.append("capture_month",$("planner-capture-month").value);form.append("confidence",$("planner-confidence").value);form.append("longitude",$("planner-upload-longitude").value.trim());form.append("latitude",$("planner-upload-latitude").value.trim());form.append("scenario_hour",$("planner-map-hour").value);form.append("generate_realistic",generateRealistic?"1":"0");form.append("manual_mask_png_base64",manualMask||"");form.append("feedback_consent",manualMask&&$("planner-feedback-consent").checked?"1":"0");
    const response = await fetch("/api/planner/analyze-upload", {method: "POST", body: form, cache: "no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error?.message || "上传图像分析失败");
    if(!generateRealistic){resetPlannerMaskEditor();state.plannerCurrentAnalysis=result;state.plannerOriginalImageUrl=URL.createObjectURL(file)}
    const guidance=`data:image/png;base64,${result.overlay_png_base64}`;resetPlannerOutputs(guidance);preview.src = guidance;
    const candidate=Boolean(result.generation_candidate_available??result.optimization_eligible),hasAutomaticRegion=Boolean((result.suggestion_regions||[]).some(item=>item.type==="canopy_target"&&Number(item.pixel_ratio||0)>0));
    const hasTeacher=Boolean(result.teacher_reference_available);
    $("generate-realistic-tree").hidden=false;$("generate-realistic-tree").disabled=!hasTeacher&&(!candidate||!hasAutomaticRegion);$("generate-realistic-tree").textContent=hasTeacher?"查看Image2教师参考":!candidate?"当前方位不建议新增遮荫":!hasAutomaticRegion?"请先微调树冠区域":"生成当前方位现实树木情景";
    $("start-mask-edit").hidden=!candidate;$("start-mask-edit").disabled=!candidate;
    showPlannerReviewControls();
    showOverlayLegend(result.overlay_legend);
    renderPublicSpaceMeter(result.public_space_score,result.space_type_assessment);
    $("planner-result-title").textContent = `上传方位图 · ${result.decision_status}`;
    const badge = $("planner-evidence-badge"); badge.textContent = "方位图Agent完成"; badge.className = "evidence-badge formal";
    const r = result.ratios,thermal=result.thermal_reference_estimate||{},reference=result.self_supervised_reference||{},weather=thermal.scenario_weather||{};
    $("planner-metrics").innerHTML = [
      metric("预计遮荫提升",Number.isFinite(Number(thermal.estimated_shade_gain))?`+${(Number(thermal.estimated_shade_gain)*100).toFixed(1)}%`:"方案后量化"),
      metric("预计Tmrt变化",Number.isFinite(Number(thermal.estimated_delta_tmrt_c))?`${Number(thermal.estimated_delta_tmrt_c).toFixed(2)}°C`:"待量化"),
      metric("预计UTCI变化",Number.isFinite(Number(thermal.estimated_delta_utci_c))?`${Number(thermal.estimated_delta_utci_c).toFixed(2)}°C`:"待量化")
    ].join("");
    const box = $("planner-recommendation"); box.hidden = false;
    const vlm=result.vlm||{}, blockers=(result.generation_blockers||[]).join("、")||"无";
    box.innerHTML = `${plannerDecisionPipeline(result)}<h3>${escapePlannerText(result.planning_priority)}</h3><p>${escapePlannerText(result.recommendation)}</p><div class="planner-brief-facts"><span>天空视域 <b>${Number(result.svf).toFixed(3)}</b></span><span>空间 <b>${escapePlannerText(result.space_type_assessment)}</b></span><span>生成 <b>${blockers==="无"?"可进入":"需复核"}</b></span></div><details><summary>查看多模态证据与情景边界</summary><p>道路尽头已排除；左右两侧独立检查慢行和树冠缺口。DINOv2最近南京形态点：${escapePlannerText(reference.nearest_morphology_point_id||"—")}。VLM：${escapePlannerText(vlm.structural_scene||"方位结构判断")}；${escapePlannerText(vlm.tree_state||"树木状态判断")}。</p><p>${weather.scenario_date||"2024-08-04"} ${weather.hour??14}:00南京晴热参考；${escapePlannerText(thermal.claim_boundary||"为情景迁移估计，不是现场实测。")} ${blockers==="无"?"自动门槛已满足。":"待复核项："+escapePlannerText(blockers)}。</p></details>`;
    renderAgentTrace(result);if(generateRealistic)return result;finishPlannerProgress("上传方位图分析完成，规划师复核已就绪");
  } catch (error) {
    if(!generateRealistic)failPlannerProgress(error.message);
    $("planner-result-title").textContent = "分析失败";
    $("planner-recommendation").hidden = false;
    $("planner-recommendation").innerHTML = `<h3>无法完成评估</h3><p>${error.message}</p>`;
  } finally { button.disabled = false; button.textContent = "确认需求并运行Agent"; }
}

async function init() {
  initializeResizablePanels();
  document.querySelectorAll(".workspace-tab").forEach(button => button.onclick = () => switchWorkspace(button.dataset.workspace));
  document.querySelectorAll(".planner-source").forEach(button => button.onclick = () => switchPlannerSource(button.dataset.source));
  $("load-planner-case").onclick = showPlannerCase;
  $("planner-intent-parse").onclick = parsePlannerIntent;
  $("planner-intent").oninput=()=>{$("planner-intent-confirmation").hidden=true};
  $("analyze-planner-upload").onclick = analyzePlannerUpload;
  $("planner-map-hour").onchange = ()=>{state.plannerDirectionBatch=null;loadPlannerPoints()};
  $("planner-confidence").oninput=event=>{
    const value=Number(event.target.value);$("planner-confidence-value").value=String(value);$("planner-confidence-value").textContent=String(value);
    clearTimeout(state.plannerConfidenceTimer);state.plannerDirectionBatch=null;if(state.selectedPlannerPoint&&state.plannerDirectionHeading!==null)state.plannerConfidenceTimer=setTimeout(()=>analyzePlannerDirection(state.selectedPlannerPoint,state.plannerDirectionHeading).catch(error=>{$("planner-direction-note").textContent=error.message}),420);
  };
  const plannerMap=$("planner-point-map");
  plannerMap.onclick=handlePlannerMapClick;
  plannerMap.onwheel=event=>{event.preventDefault();zoomPlannerMap(event.deltaY<0?1.28:1/1.28,event.clientX,event.clientY)};
  plannerMap.onpointerdown=beginPlannerMapDrag;plannerMap.onpointermove=movePlannerMap;plannerMap.onpointerup=endPlannerMapDrag;plannerMap.onpointercancel=endPlannerMapDrag;
  plannerMap.onkeydown=handlePlannerMapKey;
  $("planner-map-reset").onclick=()=>resetPlannerMapView();
  $("planner-map-zoom-in").onclick=()=>zoomPlannerMap(1.45);
  $("planner-map-zoom-out").onclick=()=>zoomPlannerMap(1/1.45);
  $("planner-show-buildings").onchange=()=>{drawPlannerMap();if($("planner-show-buildings").checked)schedulePlannerVisualRefresh()};
  $("planner-map-layer-plan").onclick=()=>setPlannerMapLayer("plan");
  $("planner-map-layer-shade").onclick=()=>setPlannerMapLayer("shade");
  $("planner-direction-view").onclick=()=>{if(state.selectedPlannerPoint&&state.plannerDirectionHeading!==null)analyzePlannerDirection(state.selectedPlannerPoint,state.plannerDirectionHeading).catch(error=>{$("planner-direction-note").textContent=error.message})};
  $("planner-fusion-view").onclick=()=>{if(state.selectedPlannerPoint)loadPlannerDirectionFusion(state.selectedPlannerPoint).catch(error=>{$("planner-direction-note").textContent=error.message})};
  $("generate-point-plan").onclick = generatePointPlan;
  $("generate-realistic-tree").onclick=generateRealisticTreeScenario;
  $("start-mask-edit").onclick=()=>initializePlannerMaskEditor().catch(error=>{const box=$("planner-recommendation");box.hidden=false;box.insertAdjacentHTML("afterbegin",`<p><strong>无法打开区域编辑器：</strong>${escapePlannerText(error.message)}</p>`) });
  $("planner-brush-add").onclick=()=>setPlannerMaskMode("add");
  $("planner-brush-erase").onclick=()=>setPlannerMaskMode("erase");
  $("planner-brush-size").oninput=event=>{$("planner-brush-size-value").value=event.target.value;$("planner-brush-size-value").textContent=event.target.value};
  $("planner-mask-undo").onclick=()=>{if(!state.plannerMaskLayer||!state.plannerMaskHistory.length)return;state.plannerMaskLayer.getContext("2d").putImageData(state.plannerMaskHistory.pop(),0,0);state.plannerMaskDirty=true;state.plannerCommittedMask=null;$("planner-mask-submit").disabled=false;$("planner-editor-status").textContent="已撤销一步 · 修改尚未提交";renderPlannerMaskEditor()};
  $("planner-mask-reset").onclick=()=>{if(!state.plannerMaskLayer||!state.plannerMaskAutomatic)return;plannerMaskSnapshot();state.plannerMaskLayer.getContext("2d").putImageData(state.plannerMaskAutomatic,0,0);state.plannerMaskDirty=false;state.plannerCommittedMask=null;$("planner-mask-submit").disabled=true;$("planner-editor-status").textContent="已恢复自动区域";$("generate-realistic-tree").textContent="生成当前方位现实树木情景";renderPlannerMaskEditor();updatePlannerFeedbackStatus()};
  $("planner-mask-submit").onclick=submitPlannerMask;
  $("planner-feedback-consent").onchange=updatePlannerFeedbackStatus;
  $("planner-no-intervention").onchange=setPlannerNoInterventionMode;
  $("planner-no-intervention-submit").onclick=submitPlannerNoIntervention;
  const maskCanvas=$("planner-mask-canvas");
  maskCanvas.onpointerdown=event=>{if(!state.plannerMaskLayer)return;event.preventDefault();plannerMaskSnapshot();state.plannerMaskDrawing=true;state.plannerMaskLastPoint=null;maskCanvas.setPointerCapture(event.pointerId);updatePlannerBrushCursor(event);paintPlannerMask(event)};
  maskCanvas.onpointermove=event=>{updatePlannerBrushCursor(event);if(state.plannerMaskDrawing){event.preventDefault();paintPlannerMask(event)}};
  maskCanvas.onpointerenter=updatePlannerBrushCursor;maskCanvas.onpointerleave=()=>{$("planner-brush-cursor").hidden=true};
  const endMaskStroke=event=>{state.plannerMaskDrawing=false;state.plannerMaskLastPoint=null;if(maskCanvas.hasPointerCapture?.(event.pointerId))maskCanvas.releasePointerCapture(event.pointerId)};
  maskCanvas.onpointerup=endMaskStroke;maskCanvas.onpointercancel=endMaskStroke;
  for(let h=6;h<=18;h++){const o=document.createElement("option");o.value=h;o.textContent=`${String(h).padStart(2,"0")}:00`;o.selected=h===14;$("hour").appendChild(o)}
  $("origin-search").onclick=()=>searchPlace("origin"); $("destination-search").onclick=()=>searchPlace("destination");
  $("origin-map-pick").onclick=()=>setMapPick("origin"); $("destination-map-pick").onclick=()=>setMapPick("destination");
  $("origin-query").onkeydown=e=>{if(e.key==="Enter")searchPlace("origin")}; $("destination-query").onkeydown=e=>{if(e.key==="Enter")searchPlace("destination")};
  $("single-route").onclick=()=>calculate(false); $("compare-routes").onclick=()=>calculate(true);
  $("assistant-parse").onclick=parseAssistantRequest;
  $("assistant-input").onkeydown=e=>{if(e.key==="Enter"&&(e.ctrlKey||e.metaKey))parseAssistantRequest()};
  $("assistant-explain").onclick=explainRoutes;
  $("clear-request").onclick=clearRequest; $("export-route").onclick=exportSelected;
  $("clear-origin").onclick=()=>clearPlace("origin");
  $("clear-destination").onclick=()=>clearPlace("destination");
  $("fit-routes").onclick=()=>{state.fitRoutesOnly=false;state.mapZoom=1;state.mapPanX=0;state.mapPanY=0;drawMap();scheduleVisualLayerRefresh()};
  $("show-buildings").onchange=drawMap;
  $("show-shadows").onchange=drawMap;
  $("zoom-in").onclick=()=>zoomMap(1.4);
  $("zoom-out").onclick=()=>zoomMap(1/1.4);
  $("detour").oninput=()=>{$("detour-value").textContent=`${$("detour").value}%`};
  $("objective").onchange=()=>{$("objective-help").textContent=OBJECTIVE_HELP[$("objective").value]};
  $("hour").onchange=()=>{
    if(state.context?.bbox) loadVisualLayers(true,[...state.context.bbox]).then(()=>scheduleVisualLayerRefresh());
    if(state.routes.length){
      showWarnings(["出发时间已变更：全域阴影正在更新；请重新计算路线以同步路线成本。"]);
    }
  };
  window.onresize=()=>{drawMap();drawPlannerMap();scheduleVisualLayerRefresh();schedulePlannerVisualRefresh()};
  const routeMap=$("route-map");
  routeMap.onwheel=event=>{event.preventDefault();zoomMap(event.deltaY<0?1.22:1/1.22,event.clientX,event.clientY)};
  routeMap.onpointerdown=beginMapDrag;routeMap.onpointermove=moveMap;
  routeMap.onpointerup=endMapDrag;routeMap.onpointercancel=endMapDrag;
  routeMap.onkeydown=handleMapKey;
  routeMap.onclick=event=>handleMapPick(event);
  try {
    const [health, context]=await Promise.all([api("/api/health"), fetch("/assets/local_context.json",{cache:"no-store"}).then(r=>r.ok?r.json():null), loadPlannerCases()]);
    state.context=context;
    if(state.context){
      let contextMinX=Infinity,contextMinY=Infinity,contextMaxX=-Infinity,contextMaxY=-Infinity;
      state.context.features.forEach(f=>{
        const line=f.geometry.coordinates;
        let minx=Infinity,miny=Infinity,maxx=-Infinity,maxy=-Infinity;
        line.forEach(p=>{minx=Math.min(minx,p[0]);miny=Math.min(miny,p[1]);maxx=Math.max(maxx,p[0]);maxy=Math.max(maxy,p[1])});
        f._bbox=[minx,miny,maxx,maxy];
        contextMinX=Math.min(contextMinX,minx);contextMinY=Math.min(contextMinY,miny);
        contextMaxX=Math.max(contextMaxX,maxx);contextMaxY=Math.max(contextMaxY,maxy);
      });
      state.context.bbox=state.context.bbox||[contextMinX,contextMinY,contextMaxX,contextMaxY];
    }
    $("health-status").classList.add("ready"); $("health-status").lastChild.textContent="本地引擎已就绪";
    const assistant=health.local_assistant;
    $("assistant-badge").textContent=assistant?.available?"Qwen3 · GPU":"模型未就绪";
    $("assistant-badge").classList.add(assistant?.available?"ready":"offline");
    drawMap();
    loadVisualLayers(false,[...state.context.bbox]);
  } catch(error) { $("health-status").lastChild.textContent="本地引擎不可用"; showWarnings([error.message]); }
}

function setMapPick(kind){
  state.mapPickTarget=kind;
  $("route-map").classList.add("picking");
  document.querySelectorAll(".map-pick-button").forEach(button=>button.classList.remove("active"));
  $(`${kind}-map-pick`).classList.add("active");
  $("map-message").hidden=false;
  $("map-message").textContent=`请在本地道路底图上点选${kind==="origin"?"起点":"终点"}`;
}

function handleMapPick(event){
  if(state.mapDragged){state.mapDragged=false;return}
  if(!state.mapPickTarget || !state.mapTransform) return;
  const t=state.mapTransform, rect=$("route-map").getBoundingClientRect();
  const sx=event.clientX-rect.left, sy=event.clientY-rect.top;
  const [x,y]=t.unproject(sx,sy);
  const kind=state.mapPickTarget;
  state[kind]={x,y,display_name:"地图点选位置"};
  $(`${kind}-query`).value="地图点选位置";
  $(`${kind}-candidates`).innerHTML=`<button class="candidate selected">地图点选位置<small>EPSG:32650 · 仅在内存中使用</small></button>`;
  $(`${kind}-map-pick`).classList.remove("active");
  $("route-map").classList.remove("picking");
  state.mapPickTarget=null;
  $("map-message").hidden=true;
  validateSnap(false);
  drawMap();
}
init();
