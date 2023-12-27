(function(){'use strict';var r;function aa(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}}
var ba="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;a[b]=c.value;return a};
function ca(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("Cannot find global object");}
var da=ca(this);function v(a,b){if(b)a:{var c=da;a=a.split(".");for(var d=0;d<a.length-1;d++){var e=a[d];if(!(e in c))break a;c=c[e]}a=a[a.length-1];d=c[a];b=b(d);b!=d&&null!=b&&ba(c,a,{configurable:!0,writable:!0,value:b})}}
v("Symbol",function(a){function b(f){if(this instanceof b)throw new TypeError("Symbol is not a constructor");return new c(d+(f||"")+"_"+e++,f)}
function c(f,g){this.g=f;ba(this,"description",{configurable:!0,writable:!0,value:g})}
if(a)return a;c.prototype.toString=function(){return this.g};
var d="jscomp_symbol_"+(1E9*Math.random()>>>0)+"_",e=0;return b});
v("Symbol.iterator",function(a){if(a)return a;a=Symbol("Symbol.iterator");for(var b="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),c=0;c<b.length;c++){var d=da[b[c]];"function"===typeof d&&"function"!=typeof d.prototype[a]&&ba(d.prototype,a,{configurable:!0,writable:!0,value:function(){return ea(aa(this))}})}return a});
function ea(a){a={next:a};a[Symbol.iterator]=function(){return this};
return a}
function w(a){var b="undefined"!=typeof Symbol&&Symbol.iterator&&a[Symbol.iterator];if(b)return b.call(a);if("number"==typeof a.length)return{next:aa(a)};throw Error(String(a)+" is not an iterable or ArrayLike");}
function x(a){if(!(a instanceof Array)){a=w(a);for(var b,c=[];!(b=a.next()).done;)c.push(b.value);a=c}return a}
function fa(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
var ha="function"==typeof Object.assign?Object.assign:function(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(d)for(var e in d)fa(d,e)&&(a[e]=d[e])}return a};
v("Object.assign",function(a){return a||ha});
var ka="function"==typeof Object.create?Object.create:function(a){function b(){}
b.prototype=a;return new b},la;
if("function"==typeof Object.setPrototypeOf)la=Object.setPrototypeOf;else{var ma;a:{var na={a:!0},oa={};try{oa.__proto__=na;ma=oa.a;break a}catch(a){}ma=!1}la=ma?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var pa=la;
function y(a,b){a.prototype=ka(b.prototype);a.prototype.constructor=a;if(pa)pa(a,b);else for(var c in b)if("prototype"!=c)if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.na=b.prototype}
function qa(){this.G=!1;this.l=null;this.h=void 0;this.g=1;this.u=this.o=0;this.O=this.i=null}
function ta(a){if(a.G)throw new TypeError("Generator is already running");a.G=!0}
qa.prototype.K=function(a){this.h=a};
function ua(a,b){a.i={mc:b,xc:!0};a.g=a.o||a.u}
qa.prototype.return=function(a){this.i={return:a};this.g=this.u};
function A(a,b,c){a.g=c;return{value:b}}
qa.prototype.B=function(a){this.g=a};
function va(a,b,c){a.o=b;void 0!=c&&(a.u=c)}
function wa(a){a.o=0;var b=a.i.mc;a.i=null;return b}
function xa(a){var b=a.O.splice(0)[0];(b=a.i=a.i||b)?b.xc?a.g=a.o||a.u:void 0!=b.B&&a.u<b.B?(a.g=b.B,a.i=null):a.g=a.u:a.g=0}
function ya(a){this.g=new qa;this.h=a}
function za(a,b){ta(a.g);var c=a.g.l;if(c)return Aa(a,"return"in c?c["return"]:function(d){return{value:d,done:!0}},b,a.g.return);
a.g.return(b);return Ba(a)}
function Aa(a,b,c,d){try{var e=b.call(a.g.l,c);if(!(e instanceof Object))throw new TypeError("Iterator result "+e+" is not an object");if(!e.done)return a.g.G=!1,e;var f=e.value}catch(g){return a.g.l=null,ua(a.g,g),Ba(a)}a.g.l=null;d.call(a.g,f);return Ba(a)}
function Ba(a){for(;a.g.g;)try{var b=a.h(a.g);if(b)return a.g.G=!1,{value:b.value,done:!1}}catch(c){a.g.h=void 0,ua(a.g,c)}a.g.G=!1;if(a.g.i){b=a.g.i;a.g.i=null;if(b.xc)throw b.mc;return{value:b.return,done:!0}}return{value:void 0,done:!0}}
function Da(a){this.next=function(b){ta(a.g);a.g.l?b=Aa(a,a.g.l.next,b,a.g.K):(a.g.K(b),b=Ba(a));return b};
this.throw=function(b){ta(a.g);a.g.l?b=Aa(a,a.g.l["throw"],b,a.g.K):(ua(a.g,b),b=Ba(a));return b};
this.return=function(b){return za(a,b)};
this[Symbol.iterator]=function(){return this}}
function Ea(a){function b(d){return a.next(d)}
function c(d){return a.throw(d)}
return new Promise(function(d,e){function f(g){g.done?d(g.value):Promise.resolve(g.value).then(b,c).then(f,e)}
f(a.next())})}
function B(a){return Ea(new Da(new ya(a)))}
function Ga(){for(var a=Number(this),b=[],c=a;c<arguments.length;c++)b[c-a]=arguments[c];return b}
v("Reflect.setPrototypeOf",function(a){return a?a:pa?function(b,c){try{return pa(b,c),!0}catch(d){return!1}}:null});
v("Promise",function(a){function b(g){this.g=0;this.i=void 0;this.h=[];this.G=!1;var h=this.l();try{g(h.resolve,h.reject)}catch(k){h.reject(k)}}
function c(){this.g=null}
function d(g){return g instanceof b?g:new b(function(h){h(g)})}
if(a)return a;c.prototype.h=function(g){if(null==this.g){this.g=[];var h=this;this.i(function(){h.o()})}this.g.push(g)};
var e=da.setTimeout;c.prototype.i=function(g){e(g,0)};
c.prototype.o=function(){for(;this.g&&this.g.length;){var g=this.g;this.g=[];for(var h=0;h<g.length;++h){var k=g[h];g[h]=null;try{k()}catch(l){this.l(l)}}}this.g=null};
c.prototype.l=function(g){this.i(function(){throw g;})};
b.prototype.l=function(){function g(l){return function(n){k||(k=!0,l.call(h,n))}}
var h=this,k=!1;return{resolve:g(this.cb),reject:g(this.o)}};
b.prototype.cb=function(g){if(g===this)this.o(new TypeError("A Promise cannot resolve to itself"));else if(g instanceof b)this.fb(g);else{a:switch(typeof g){case "object":var h=null!=g;break a;case "function":h=!0;break a;default:h=!1}h?this.ya(g):this.u(g)}};
b.prototype.ya=function(g){var h=void 0;try{h=g.then}catch(k){this.o(k);return}"function"==typeof h?this.sb(h,g):this.u(g)};
b.prototype.o=function(g){this.K(2,g)};
b.prototype.u=function(g){this.K(1,g)};
b.prototype.K=function(g,h){if(0!=this.g)throw Error("Cannot settle("+g+", "+h+"): Promise already settled in state"+this.g);this.g=g;this.i=h;2===this.g&&this.eb();this.O()};
b.prototype.eb=function(){var g=this;e(function(){if(g.oa()){var h=da.console;"undefined"!==typeof h&&h.error(g.i)}},1)};
b.prototype.oa=function(){if(this.G)return!1;var g=da.CustomEvent,h=da.Event,k=da.dispatchEvent;if("undefined"===typeof k)return!0;"function"===typeof g?g=new g("unhandledrejection",{cancelable:!0}):"function"===typeof h?g=new h("unhandledrejection",{cancelable:!0}):(g=da.document.createEvent("CustomEvent"),g.initCustomEvent("unhandledrejection",!1,!0,g));g.promise=this;g.reason=this.i;return k(g)};
b.prototype.O=function(){if(null!=this.h){for(var g=0;g<this.h.length;++g)f.h(this.h[g]);this.h=null}};
var f=new c;b.prototype.fb=function(g){var h=this.l();g.wb(h.resolve,h.reject)};
b.prototype.sb=function(g,h){var k=this.l();try{g.call(h,k.resolve,k.reject)}catch(l){k.reject(l)}};
b.prototype.then=function(g,h){function k(q,p){return"function"==typeof q?function(t){try{l(q(t))}catch(u){n(u)}}:p}
var l,n,m=new b(function(q,p){l=q;n=p});
this.wb(k(g,l),k(h,n));return m};
b.prototype.catch=function(g){return this.then(void 0,g)};
b.prototype.wb=function(g,h){function k(){switch(l.g){case 1:g(l.i);break;case 2:h(l.i);break;default:throw Error("Unexpected state: "+l.g);}}
var l=this;null==this.h?f.h(k):this.h.push(k);this.G=!0};
b.resolve=d;b.reject=function(g){return new b(function(h,k){k(g)})};
b.race=function(g){return new b(function(h,k){for(var l=w(g),n=l.next();!n.done;n=l.next())d(n.value).wb(h,k)})};
b.all=function(g){var h=w(g),k=h.next();return k.done?d([]):new b(function(l,n){function m(t){return function(u){q[t]=u;p--;0==p&&l(q)}}
var q=[],p=0;do q.push(void 0),p++,d(k.value).wb(m(q.length-1),n),k=h.next();while(!k.done)})};
return b});
v("WeakMap",function(a){function b(k){this.g=(h+=Math.random()+1).toString();if(k){k=w(k);for(var l;!(l=k.next()).done;)l=l.value,this.set(l[0],l[1])}}
function c(){}
function d(k){var l=typeof k;return"object"===l&&null!==k||"function"===l}
function e(k){if(!fa(k,g)){var l=new c;ba(k,g,{value:l})}}
function f(k){var l=Object[k];l&&(Object[k]=function(n){if(n instanceof c)return n;Object.isExtensible(n)&&e(n);return l(n)})}
if(function(){if(!a||!Object.seal)return!1;try{var k=Object.seal({}),l=Object.seal({}),n=new a([[k,2],[l,3]]);if(2!=n.get(k)||3!=n.get(l))return!1;n.delete(k);n.set(l,4);return!n.has(k)&&4==n.get(l)}catch(m){return!1}}())return a;
var g="$jscomp_hidden_"+Math.random();f("freeze");f("preventExtensions");f("seal");var h=0;b.prototype.set=function(k,l){if(!d(k))throw Error("Invalid WeakMap key");e(k);if(!fa(k,g))throw Error("WeakMap key fail: "+k);k[g][this.g]=l;return this};
b.prototype.get=function(k){return d(k)&&fa(k,g)?k[g][this.g]:void 0};
b.prototype.has=function(k){return d(k)&&fa(k,g)&&fa(k[g],this.g)};
b.prototype.delete=function(k){return d(k)&&fa(k,g)&&fa(k[g],this.g)?delete k[g][this.g]:!1};
return b});
v("Map",function(a){function b(){var h={};return h.previous=h.next=h.head=h}
function c(h,k){var l=h[1];return ea(function(){if(l){for(;l.head!=h[1];)l=l.previous;for(;l.next!=l.head;)return l=l.next,{done:!1,value:k(l)};l=null}return{done:!0,value:void 0}})}
function d(h,k){var l=k&&typeof k;"object"==l||"function"==l?f.has(k)?l=f.get(k):(l=""+ ++g,f.set(k,l)):l="p_"+k;var n=h[0][l];if(n&&fa(h[0],l))for(h=0;h<n.length;h++){var m=n[h];if(k!==k&&m.key!==m.key||k===m.key)return{id:l,list:n,index:h,entry:m}}return{id:l,list:n,index:-1,entry:void 0}}
function e(h){this[0]={};this[1]=b();this.size=0;if(h){h=w(h);for(var k;!(k=h.next()).done;)k=k.value,this.set(k[0],k[1])}}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var h=Object.seal({x:4}),k=new a(w([[h,"s"]]));if("s"!=k.get(h)||1!=k.size||k.get({x:4})||k.set({x:4},"t")!=k||2!=k.size)return!1;var l=k.entries(),n=l.next();if(n.done||n.value[0]!=h||"s"!=n.value[1])return!1;n=l.next();return n.done||4!=n.value[0].x||"t"!=n.value[1]||!l.next().done?!1:!0}catch(m){return!1}}())return a;
var f=new WeakMap;e.prototype.set=function(h,k){h=0===h?0:h;var l=d(this,h);l.list||(l.list=this[0][l.id]=[]);l.entry?l.entry.value=k:(l.entry={next:this[1],previous:this[1].previous,head:this[1],key:h,value:k},l.list.push(l.entry),this[1].previous.next=l.entry,this[1].previous=l.entry,this.size++);return this};
e.prototype.delete=function(h){h=d(this,h);return h.entry&&h.list?(h.list.splice(h.index,1),h.list.length||delete this[0][h.id],h.entry.previous.next=h.entry.next,h.entry.next.previous=h.entry.previous,h.entry.head=null,this.size--,!0):!1};
e.prototype.clear=function(){this[0]={};this[1]=this[1].previous=b();this.size=0};
e.prototype.has=function(h){return!!d(this,h).entry};
e.prototype.get=function(h){return(h=d(this,h).entry)&&h.value};
e.prototype.entries=function(){return c(this,function(h){return[h.key,h.value]})};
e.prototype.keys=function(){return c(this,function(h){return h.key})};
e.prototype.values=function(){return c(this,function(h){return h.value})};
e.prototype.forEach=function(h,k){for(var l=this.entries(),n;!(n=l.next()).done;)n=n.value,h.call(k,n[1],n[0],this)};
e.prototype[Symbol.iterator]=e.prototype.entries;var g=0;return e});
function Ha(a,b,c){if(null==a)throw new TypeError("The 'this' value for String.prototype."+c+" must not be null or undefined");if(b instanceof RegExp)throw new TypeError("First argument to String.prototype."+c+" must not be a regular expression");return a+""}
v("String.prototype.endsWith",function(a){return a?a:function(b,c){var d=Ha(this,b,"endsWith");b+="";void 0===c&&(c=d.length);c=Math.max(0,Math.min(c|0,d.length));for(var e=b.length;0<e&&0<c;)if(d[--c]!=b[--e])return!1;return 0>=e}});
v("Object.setPrototypeOf",function(a){return a||pa});
v("Array.prototype.find",function(a){return a?a:function(b,c){a:{var d=this;d instanceof String&&(d=String(d));for(var e=d.length,f=0;f<e;f++){var g=d[f];if(b.call(c,g,f,d)){b=g;break a}}b=void 0}return b}});
v("String.prototype.startsWith",function(a){return a?a:function(b,c){var d=Ha(this,b,"startsWith");b+="";var e=d.length,f=b.length;c=Math.max(0,Math.min(c|0,d.length));for(var g=0;g<f&&c<e;)if(d[c++]!=b[g++])return!1;return g>=f}});
v("Number.isFinite",function(a){return a?a:function(b){return"number"!==typeof b?!1:!isNaN(b)&&Infinity!==b&&-Infinity!==b}});
v("Math.trunc",function(a){return a?a:function(b){b=Number(b);if(isNaN(b)||Infinity===b||-Infinity===b||0===b)return b;var c=Math.floor(Math.abs(b));return 0>b?-c:c}});
v("Object.values",function(a){return a?a:function(b){var c=[],d;for(d in b)fa(b,d)&&c.push(b[d]);return c}});
v("Object.is",function(a){return a?a:function(b,c){return b===c?0!==b||1/b===1/c:b!==b&&c!==c}});
v("Array.prototype.includes",function(a){return a?a:function(b,c){var d=this;d instanceof String&&(d=String(d));var e=d.length;c=c||0;for(0>c&&(c=Math.max(c+e,0));c<e;c++){var f=d[c];if(f===b||Object.is(f,b))return!0}return!1}});
v("String.prototype.includes",function(a){return a?a:function(b,c){return-1!==Ha(this,b,"includes").indexOf(b,c||0)}});
v("Set",function(a){function b(c){this.g=new Map;if(c){c=w(c);for(var d;!(d=c.next()).done;)this.add(d.value)}this.size=this.g.size}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var c=Object.seal({x:4}),d=new a(w([c]));if(!d.has(c)||1!=d.size||d.add(c)!=d||1!=d.size||d.add({x:4})!=d||2!=d.size)return!1;var e=d.entries(),f=e.next();if(f.done||f.value[0]!=c||f.value[1]!=c)return!1;f=e.next();return f.done||f.value[0]==c||4!=f.value[0].x||f.value[1]!=f.value[0]?!1:e.next().done}catch(g){return!1}}())return a;
b.prototype.add=function(c){c=0===c?0:c;this.g.set(c,c);this.size=this.g.size;return this};
b.prototype.delete=function(c){c=this.g.delete(c);this.size=this.g.size;return c};
b.prototype.clear=function(){this.g.clear();this.size=0};
b.prototype.has=function(c){return this.g.has(c)};
b.prototype.entries=function(){return this.g.entries()};
b.prototype.values=function(){return this.g.values()};
b.prototype.keys=b.prototype.values;b.prototype[Symbol.iterator]=b.prototype.values;b.prototype.forEach=function(c,d){var e=this;this.g.forEach(function(f){return c.call(d,f,f,e)})};
return b});
v("Number.MAX_SAFE_INTEGER",function(){return 9007199254740991});
v("Number.isInteger",function(a){return a?a:function(b){return Number.isFinite(b)?b===Math.floor(b):!1}});
v("Number.isSafeInteger",function(a){return a?a:function(b){return Number.isInteger(b)&&Math.abs(b)<=Number.MAX_SAFE_INTEGER}});
v("Number.isNaN",function(a){return a?a:function(b){return"number"===typeof b&&isNaN(b)}});
function Ia(a,b){a instanceof String&&(a+="");var c=0,d=!1,e={next:function(){if(!d&&c<a.length){var f=c++;return{value:b(f,a[f]),done:!1}}d=!0;return{done:!0,value:void 0}}};
e[Symbol.iterator]=function(){return e};
return e}
v("Array.prototype.entries",function(a){return a?a:function(){return Ia(this,function(b,c){return[b,c]})}});
v("Array.prototype.keys",function(a){return a?a:function(){return Ia(this,function(b){return b})}});
v("Array.prototype.values",function(a){return a?a:function(){return Ia(this,function(b,c){return c})}});
v("Array.from",function(a){return a?a:function(b,c,d){c=null!=c?c:function(h){return h};
var e=[],f="undefined"!=typeof Symbol&&Symbol.iterator&&b[Symbol.iterator];if("function"==typeof f){b=f.call(b);for(var g=0;!(f=b.next()).done;)e.push(c.call(d,f.value,g++))}else for(f=b.length,g=0;g<f;g++)e.push(c.call(d,b[g],g));return e}});
v("Object.entries",function(a){return a?a:function(b){var c=[],d;for(d in b)fa(b,d)&&c.push([d,b[d]]);return c}});/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
var D=this||self;function Ja(a,b){var c=E("CLOSURE_FLAGS");a=c&&c[a];return null!=a?a:b}
function E(a,b){a=a.split(".");b=b||D;for(var c=0;c<a.length;c++)if(b=b[a[c]],null==b)return null;return b}
function Ka(a){var b=typeof a;return"object"!=b?b:a?Array.isArray(a)?"array":b:"null"}
function Na(a){var b=Ka(a);return"array"==b||"object"==b&&"number"==typeof a.length}
function Oa(a){var b=typeof a;return"object"==b&&null!=a||"function"==b}
function Qa(a){return Object.prototype.hasOwnProperty.call(a,Ra)&&a[Ra]||(a[Ra]=++Sa)}
var Ra="closure_uid_"+(1E9*Math.random()>>>0),Sa=0;function Ta(a,b,c){return a.call.apply(a.bind,arguments)}
function Ua(a,b,c){if(!a)throw Error();if(2<arguments.length){var d=Array.prototype.slice.call(arguments,2);return function(){var e=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(e,d);return a.apply(b,e)}}return function(){return a.apply(b,arguments)}}
function Va(a,b,c){Va=Function.prototype.bind&&-1!=Function.prototype.bind.toString().indexOf("native code")?Ta:Ua;return Va.apply(null,arguments)}
function Wa(){return Date.now()}
function F(a,b){a=a.split(".");var c=D;a[0]in c||"undefined"==typeof c.execScript||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)a.length||void 0===b?c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}:c[d]=b}
function Xa(a,b){function c(){}
c.prototype=b.prototype;a.na=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.pe=function(d,e,f){for(var g=Array(arguments.length-2),h=2;h<arguments.length;h++)g[h-2]=arguments[h];return b.prototype[e].apply(d,g)}}
function Ya(a){return a}
;function Za(a,b){if(Error.captureStackTrace)Error.captureStackTrace(this,Za);else{var c=Error().stack;c&&(this.stack=c)}a&&(this.message=String(a));void 0!==b&&(this.cause=b)}
Xa(Za,Error);Za.prototype.name="CustomError";function $a(a){a=a.url;var b=/[?&]dsh=1(&|$)/.test(a);this.i=!b&&/[?&]ae=1(&|$)/.test(a);this.l=!b&&/[?&]ae=2(&|$)/.test(a);if((this.g=/[?&]adurl=([^&]*)/.exec(a))&&this.g[1]){try{var c=decodeURIComponent(this.g[1])}catch(d){c=null}this.h=c}}
;function ab(){}
function bb(a){var b=!1,c;return function(){b||(c=a(),b=!0);return c}}
;var cb=Array.prototype.indexOf?function(a,b){return Array.prototype.indexOf.call(a,b,void 0)}:function(a,b){if("string"===typeof a)return"string"!==typeof b||1!=b.length?-1:a.indexOf(b,0);
for(var c=0;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},db=Array.prototype.forEach?function(a,b,c){Array.prototype.forEach.call(a,b,c)}:function(a,b,c){for(var d=a.length,e="string"===typeof a?a.split(""):a,f=0;f<d;f++)f in e&&b.call(c,e[f],f,a)};
function eb(a,b){b=cb(a,b);var c;(c=0<=b)&&Array.prototype.splice.call(a,b,1);return c}
function fb(a){return Array.prototype.concat.apply([],arguments)}
function gb(a){var b=a.length;if(0<b){for(var c=Array(b),d=0;d<b;d++)c[d]=a[d];return c}return[]}
function kb(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(Na(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
;function lb(a,b){for(var c in a)b.call(void 0,a[c],c,a)}
function mb(a){var b=nb,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function ob(a,b){for(var c in a)if(!(c in b)||a[c]!==b[c])return!1;for(var d in b)if(!(d in a))return!1;return!0}
function pb(a){if(!a||"object"!==typeof a)return a;if("function"===typeof a.clone)return a.clone();if("undefined"!==typeof Map&&a instanceof Map)return new Map(a);if("undefined"!==typeof Set&&a instanceof Set)return new Set(a);if(a instanceof Date)return new Date(a.getTime());var b=Array.isArray(a)?[]:"function"!==typeof ArrayBuffer||"function"!==typeof ArrayBuffer.isView||!ArrayBuffer.isView(a)||a instanceof DataView?{}:new a.constructor(a.length),c;for(c in a)b[c]=pb(a[c]);return b}
var qb="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function rb(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<qb.length;f++)c=qb[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;var sb;function tb(){if(void 0===sb){var a=null,b=D.trustedTypes;if(b&&b.createPolicy){try{a=b.createPolicy("goog#html",{createHTML:Ya,createScript:Ya,createScriptURL:Ya})}catch(c){D.console&&D.console.error(c.message)}sb=a}else sb=a}return sb}
;function ub(a,b){this.g=a===vb&&b||""}
ub.prototype.toString=function(){return this.g};
function wb(a){return new ub(vb,a)}
var vb={};wb("");function xb(a){this.g=a}
xb.prototype.toString=function(){return this.g+""};
var yb={};var zb=String.prototype.trim?function(a){return a.trim()}:function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]},Ab=/&/g,Bb=/</g,Cb=/>/g,Db=/"/g,Eb=/'/g,Fb=/\x00/g,Gb=/[\x00&<>"']/;function Hb(a){this.g=a}
Hb.prototype.toString=function(){return this.g.toString()};
var Ib={},Jb=new Hb("about:invalid#zClosurez",Ib);var Kb=Ja(610401301,!1),Lb=Ja(572417392,!0);function Mb(){var a=D.navigator;return a&&(a=a.userAgent)?a:""}
var Nb,Ob=D.navigator;Nb=Ob?Ob.userAgentData||null:null;function Pb(a){return Kb?Nb?Nb.brands.some(function(b){return(b=b.brand)&&-1!=b.indexOf(a)}):!1:!1}
function G(a){return-1!=Mb().indexOf(a)}
;function Qb(){return Kb?!!Nb&&0<Nb.brands.length:!1}
function Rb(){return Qb()?!1:G("Trident")||G("MSIE")}
function Sb(){return Qb()?Pb("Chromium"):(G("Chrome")||G("CriOS"))&&!(Qb()?0:G("Edge"))||G("Silk")}
;function Yb(a){this.g=a}
Yb.prototype.toString=function(){return this.g.toString()};function Zb(a){Gb.test(a)&&(-1!=a.indexOf("&")&&(a=a.replace(Ab,"&amp;")),-1!=a.indexOf("<")&&(a=a.replace(Bb,"&lt;")),-1!=a.indexOf(">")&&(a=a.replace(Cb,"&gt;")),-1!=a.indexOf('"')&&(a=a.replace(Db,"&quot;")),-1!=a.indexOf("'")&&(a=a.replace(Eb,"&#39;")),-1!=a.indexOf("\x00")&&(a=a.replace(Fb,"&#0;")));return a}
;var $b=RegExp("^(?:([^:/?#.]+):)?(?://(?:([^\\\\/?#]*)@)?([^\\\\/?#]*?)(?::([0-9]+))?(?=[\\\\/?#]|$))?([^?#]+)?(?:\\?([^#]*))?(?:#([\\s\\S]*))?$");function ac(a){return a?decodeURI(a):a}
function bc(a){return ac(a.match($b)[3]||null)}
function cc(a){var b=a.match($b);a=b[1];var c=b[2],d=b[3];b=b[4];var e="";a&&(e+=a+":");d&&(e+="//",c&&(e+=c+"@"),e+=d,b&&(e+=":"+b));return e}
function dc(a,b,c){if(Array.isArray(b))for(var d=0;d<b.length;d++)dc(a,String(b[d]),c);else null!=b&&c.push(a+(""===b?"":"="+encodeURIComponent(String(b))))}
function ec(a){var b=[],c;for(c in a)dc(c,a[c],b);return b.join("&")}
var fc=/#|$/;function gc(a,b){var c=a.search(fc);a:{var d=0;for(var e=b.length;0<=(d=a.indexOf(b,d))&&d<c;){var f=a.charCodeAt(d-1);if(38==f||63==f)if(f=a.charCodeAt(d+e),!f||61==f||38==f||35==f)break a;d+=e+1}d=-1}if(0>d)return null;e=a.indexOf("&",d);if(0>e||e>c)e=c;d+=b.length+1;return decodeURIComponent(a.slice(d,-1!==e?e:0).replace(/\+/g," "))}
;function hc(a){D.setTimeout(function(){throw a;},0)}
;function ic(){return G("iPhone")&&!G("iPod")&&!G("iPad")}
;function jc(a){jc[" "](a);return a}
jc[" "]=function(){};var kc=Qb()?!1:G("Opera"),lc=Rb(),mc=G("Edge"),nc=G("Gecko")&&!(-1!=Mb().toLowerCase().indexOf("webkit")&&!G("Edge"))&&!(G("Trident")||G("MSIE"))&&!G("Edge"),oc=-1!=Mb().toLowerCase().indexOf("webkit")&&!G("Edge");function pc(){var a=D.document;return a?a.documentMode:void 0}
var qc;a:{var rc="",sc=function(){var a=Mb();if(nc)return/rv:([^\);]+)(\)|;)/.exec(a);if(mc)return/Edge\/([\d\.]+)/.exec(a);if(lc)return/\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/.exec(a);if(oc)return/WebKit\/(\S+)/.exec(a);if(kc)return/(?:Version)[ \/]?(\S+)/.exec(a)}();
sc&&(rc=sc?sc[1]:"");if(lc){var tc=pc();if(null!=tc&&tc>parseFloat(rc)){qc=String(tc);break a}}qc=rc}var uc=qc,vc;if(D.document&&lc){var wc=pc();vc=wc?wc:parseInt(uc,10)||void 0}else vc=void 0;var xc=vc;var yc=ic()||G("iPod"),Dc=G("iPad");!G("Android")||Sb();Sb();var Ec=G("Safari")&&!(Sb()||(Qb()?0:G("Coast"))||(Qb()?0:G("Opera"))||(Qb()?0:G("Edge"))||(Qb()?Pb("Microsoft Edge"):G("Edg/"))||(Qb()?Pb("Opera"):G("OPR"))||G("Firefox")||G("FxiOS")||G("Silk")||G("Android"))&&!(ic()||G("iPad")||G("iPod"));var Fc={},Gc=null;
function Hc(a,b){Na(a);void 0===b&&(b=0);if(!Gc){Gc={};for(var c="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""),d=["+/=","+/","-_=","-_.","-_"],e=0;5>e;e++){var f=c.concat(d[e].split(""));Fc[e]=f;for(var g=0;g<f.length;g++){var h=f[g];void 0===Gc[h]&&(Gc[h]=g)}}}b=Fc[b];c=Array(Math.floor(a.length/3));d=b[64]||"";for(e=f=0;f<a.length-2;f+=3){var k=a[f],l=a[f+1];h=a[f+2];g=b[k>>2];k=b[(k&3)<<4|l>>4];l=b[(l&15)<<2|h>>6];h=b[h&63];c[e++]=""+g+k+l+h}g=0;h=d;switch(a.length-
f){case 2:g=a[f+1],h=b[(g&15)<<2]||d;case 1:a=a[f],c[e]=""+b[a>>2]+b[(a&3)<<4|g>>4]+h+d}return c.join("")}
;var Ic="undefined"!==typeof Uint8Array,Jc=!lc&&"function"===typeof btoa;function Kc(){return"function"===typeof BigInt}
var Lc=!Lb,Mc=!Lb;var Nc=0,Oc=0;function Pc(a){var b=0>a;a=Math.abs(a);var c=a>>>0;a=Math.floor((a-c)/4294967296);b&&(c=w(Qc(c,a)),b=c.next().value,a=c.next().value,c=b);Nc=c>>>0;Oc=a>>>0}
function Rc(a,b){b>>>=0;a>>>=0;if(2097151>=b)var c=""+(4294967296*b+a);else Kc()?c=""+(BigInt(b)<<BigInt(32)|BigInt(a)):(c=(a>>>24|b<<8)&16777215,b=b>>16&65535,a=(a&16777215)+6777216*c+6710656*b,c+=8147497*b,b*=2,1E7<=a&&(c+=Math.floor(a/1E7),a%=1E7),1E7<=c&&(b+=Math.floor(c/1E7),c%=1E7),c=b+Sc(c)+Sc(a));return c}
function Sc(a){a=String(a);return"0000000".slice(a.length)+a}
function Tc(){var a=Nc,b=Oc;b&2147483648?Kc()?a=""+(BigInt(b|0)<<BigInt(32)|BigInt(a>>>0)):(b=w(Qc(a,b)),a=b.next().value,b=b.next().value,a="-"+Rc(a,b)):a=Rc(a,b);return a}
function Qc(a,b){b=~b;a?a=~a+1:b+=1;return[a,b]}
;function Uc(a){return Array.prototype.slice.call(a)}
;var Vc;Vc="function"===typeof Symbol&&"symbol"===typeof Symbol()?Symbol():void 0;Math.max.apply(Math,x(Object.values({be:1,Zd:2,Yd:4,ee:8,de:16,ce:32,Pd:64,ge:128,Wd:256,Vd:512,ae:1024,Td:2048,fe:4096,Ud:8192})));var Wc=Vc?function(a,b){a[Vc]|=b}:function(a,b){void 0!==a.ga?a.ga|=b:Object.defineProperties(a,{ga:{value:b,
configurable:!0,writable:!0,enumerable:!1}})};
function Xc(a){var b=Yc(a);1!==(b&1)&&(Object.isFrozen(a)&&(a=Uc(a)),H(a,b|1))}
var Zc=Vc?function(a,b){a[Vc]&=~b}:function(a,b){void 0!==a.ga&&(a.ga&=~b)};
function $c(a,b,c){return c?a|b:a&~b}
var Yc=Vc?function(a){return a[Vc]|0}:function(a){return a.ga|0},ad=Vc?function(a){return a[Vc]}:function(a){return a.ga},H=Vc?function(a,b){a[Vc]=b}:function(a,b){void 0!==a.ga?a.ga=b:Object.defineProperties(a,{ga:{value:b,
configurable:!0,writable:!0,enumerable:!1}})};
function bd(){var a=[];Wc(a,1);return a}
function cd(a,b){H(b,(a|0)&-14591)}
function dd(a,b){H(b,(a|34)&-14557)}
function ed(a){a=a>>14&1023;return 0===a?536870912:a}
;var fd={},gd={};function hd(a){return!(!a||"object"!==typeof a||a.te!==gd)}
function id(a){return null!==a&&"object"===typeof a&&!Array.isArray(a)&&a.constructor===Object}
var jd,kd=!Lb;function ld(a,b,c){if(!Array.isArray(a)||a.length)return!1;var d=Yc(a);if(d&1)return!0;if(!(b&&(Array.isArray(b)?b.includes(c):b.has(c))))return!1;H(a,d|1);return!0}
var md,nd=[];H(nd,55);md=Object.freeze(nd);function od(a){if(a&2)throw Error();}
Object.freeze(new function(){});
Object.freeze(new function(){});function pd(a){a=Error(a);a.__closure__error__context__984382||(a.__closure__error__context__984382={});a.__closure__error__context__984382.severity="warning";return a}
;function qd(a){return a.displayName||a.name||"unknown type name"}
function rd(a){if("boolean"!==typeof a)throw Error("Expected boolean but got "+Ka(a)+": "+a);return a}
var wd=/^-?([1-9][0-9]*|0)(\.[0-9]+)?$/;function xd(a){var b=typeof a;return"number"===b?Number.isFinite(a):"string"!==b?!1:wd.test(a)}
function yd(a,b){b=!!b;if(!xd(a))throw pd("int64");if("string"===typeof a)if(xd(a),b=Math.trunc(Number(a)),Number.isSafeInteger(b))a=String(b);else{if(b=a.indexOf("."),-1!==b&&(a=a.substring(0,b)),!zd(a)){if(16>a.length)Pc(Number(a));else if(Kc())a=BigInt(a),Nc=Number(a&BigInt(4294967295))>>>0,Oc=Number(a>>BigInt(32)&BigInt(4294967295));else{b=+("-"===a[0]);Oc=Nc=0;for(var c=a.length,d=0+b,e=(c-b)%6+b;e<=c;d=e,e+=6)d=Number(a.slice(d,e)),Oc*=1E6,Nc=1E6*Nc+d,4294967296<=Nc&&(Oc+=Math.trunc(Nc/4294967296),
Oc>>>=0,Nc>>>=0);b&&(b=w(Qc(Nc,Oc)),a=b.next().value,b=b.next().value,Nc=a,Oc=b)}a=Tc()}}else if(b)xd(a),a=Math.trunc(a),Number.isSafeInteger(a)?a=String(a):(b=String(a),zd(b)?a=b:(Pc(a),a=Tc()));else if(xd(a),a=Math.trunc(a),!Number.isSafeInteger(a)){Pc(a);b=Nc;c=Oc;if(a=c&2147483648)b=~b+1>>>0,c=~c>>>0,0==b&&(c=c+1>>>0);b=4294967296*c+(b>>>0);a=a?-b:b}return a}
function Ad(a){return null==a?a:yd(a)}
function zd(a){return"-"===a[0]?20>a.length?!0:20===a.length&&-922337<Number(a.substring(0,7)):19>a.length?!0:19===a.length&&922337>Number(a.substring(0,6))}
function Bd(a){if(null!=a&&"string"!==typeof a)throw Error();return a}
function Cd(a){return null==a||"string"===typeof a?a:void 0}
function Dd(a,b){if(!(a instanceof b))throw Error("Expected instanceof "+qd(b)+" but got "+(a&&qd(a.constructor)));return a}
function Ed(a,b,c){if(null!=a&&"object"===typeof a&&a.Tb===fd)return a;if(Array.isArray(a)){var d=Yc(a),e=d;0===e&&(e|=c&32);e|=c&2;e!==d&&H(a,e);return new b(a)}}
;var Fd;function I(a,b,c){null==a&&(a=Fd);Fd=void 0;if(null==a){var d=96;c?(a=[c],d|=512):a=[];b&&(d=d&-16760833|(b&1023)<<14)}else{if(!Array.isArray(a))throw Error();d=Yc(a);if(d&64)return a;d|=64;if(c&&(d|=512,c!==a[0]))throw Error();a:{c=a;var e=c.length;if(e){var f=e-1;if(id(c[f])){d|=256;b=f-(+!!(d&512)-1);if(1024<=b)throw Error();d=d&-16760833|(b&1023)<<14;break a}}if(b){b=Math.max(b,e-(+!!(d&512)-1));if(1024<b)throw Error();d=d&-16760833|(b&1023)<<14}}}H(a,d);return a}
;function Gd(a,b){return Hd(b)}
function Hd(a){switch(typeof a){case "number":return isFinite(a)?a:String(a);case "boolean":return a?1:0;case "object":if(a){if(Array.isArray(a))return kd||!ld(a,void 0,9999)?a:void 0;if(Ic&&null!=a&&a instanceof Uint8Array){if(Jc){for(var b="",c=0,d=a.length-10240;c<d;)b+=String.fromCharCode.apply(null,a.subarray(c,c+=10240));b+=String.fromCharCode.apply(null,c?a.subarray(c):a);a=btoa(b)}else a=Hc(a);return a}}}return a}
;function Id(a,b,c){a=Uc(a);var d=a.length,e=b&256?a[d-1]:void 0;d+=e?-1:0;for(b=b&512?1:0;b<d;b++)a[b]=c(a[b]);if(e){b=a[b]={};for(var f in e)b[f]=c(e[f])}return a}
function Jd(a,b,c,d,e,f){if(null!=a){if(Array.isArray(a))a=e&&0==a.length&&Yc(a)&1?void 0:f&&Yc(a)&2?a:Kd(a,b,c,void 0!==d,e,f);else if(id(a)){var g={},h;for(h in a)g[h]=Jd(a[h],b,c,d,e,f);a=g}else a=b(a,d);return a}}
function Kd(a,b,c,d,e,f){var g=d||c?Yc(a):0;d=d?!!(g&32):void 0;a=Uc(a);for(var h=0;h<a.length;h++)a[h]=Jd(a[h],b,c,d,e,f);c&&c(g,a);return a}
function Ld(a){return a.Tb===fd?a.toJSON():Hd(a)}
;function Md(a,b,c){c=void 0===c?dd:c;if(null!=a){if(Ic&&a instanceof Uint8Array)return b?a:new Uint8Array(a);if(Array.isArray(a)){var d=Yc(a);if(d&2)return a;b&&(b=0===d||!!(d&32)&&!(d&64||!(d&16)));return b?(H(a,(d|34)&-12293),a):Kd(a,Md,d&4?dd:c,!0,!1,!0)}a.Tb===fd&&(c=a.s,d=ad(c),a=d&2?a:Nd(a,c,d,!0));return a}}
function Nd(a,b,c,d){a=a.constructor;b=Od(b,c,d);Yc(b);Fd=b;b=new a(b);Fd=void 0;return b}
function Od(a,b,c){var d=c||b&2?dd:cd,e=!!(b&32);a=Id(a,b,function(f){return Md(f,e,d)});
Wc(a,32|(c?2:0));return a}
function Pd(a){var b=a.s,c=ad(b);return c&2?Nd(a,b,c,!1):a}
;function Qd(a,b){a=a.s;return Rd(a,ad(a),b)}
function Rd(a,b,c,d){if(-1===c)return null;if(c>=ed(b)){if(b&256)return a[a.length-1][c]}else{var e=a.length;if(d&&b&256&&(d=a[e-1][c],null!=d))return d;b=c+(+!!(b&512)-1);if(b<e)return a[b]}}
function Sd(a,b,c){var d=a.s,e=ad(d);od(e);Td(d,e,b,c);return a}
function Td(a,b,c,d,e){id(d);var f=ed(b);if(c>=f||e){e=b;if(b&256)f=a[a.length-1];else{if(null==d)return e;f=a[f+(+!!(b&512)-1)]={};e|=256}f[c]=d;e!==b&&H(a,e);return e}a[c+(+!!(b&512)-1)]=d;b&256&&(a=a[a.length-1],c in a&&delete a[c]);return b}
function Ud(a){return!!(2&a)&&!!(4&a)||!!(2048&a)}
function Vd(a,b,c,d){a=a.s;var e=ad(a);od(e);(c=Wd(a,e,c))&&c!==b&&null!=d&&(e=Td(a,e,c));Td(a,e,b,d)}
function Xd(a,b,c){a=a.s;return Wd(a,ad(a),b)===c?c:-1}
function Wd(a,b,c){for(var d=0,e=0;e<c.length;e++){var f=c[e];null!=Rd(a,b,f)&&(0!==d&&(b=Td(a,b,d)),d=f)}return d}
function Yd(a,b,c){var d=void 0===d?!1:d;var e=a.s;var f=ad(e),g=Rd(e,f,c,d);b=Ed(g,b,f);b!==g&&null!=b&&Td(e,f,c,b,d);e=b;if(null==e)return e;a=a.s;f=ad(a);f&2||(g=Pd(e),g!==e&&(e=g,Td(a,f,c,e,d)));return e}
function J(a,b,c,d){null!=d?Dd(d,b):d=void 0;return Sd(a,c,d)}
function Zd(a,b,c,d,e){null!=e?Dd(e,b):e=void 0;Vd(a,c,d,e)}
function $d(a,b){var c=!0;a=$c(a,2,!!(2&b));a=$c(a,32,!!(32&b)&&c);return a=$c(a,2048,!1)}
function ae(a,b,c,d){a=a.s;var e=ad(a);od(e);var f,g=!!(2&e),h=g?1:2,k=1===h;h=2===h;f&&(f=!g);g=Rd(a,e,b);g=Array.isArray(g)?g:md;var l=Yc(g),n=!!(4&l);if(!n){var m=l;0===m&&(m=$d(m,e));m=$c(m,1,!0);l=g;var q=e,p=!!(2&m);p&&(q=$c(q,2,!0));for(var t=!p,u=!0,z=0,C=0;z<l.length;z++){var P=Ed(l[z],c,q);if(P instanceof c){if(!p){var U=!!(Yc(P.s)&2);t&&(t=!U);u&&(u=U)}l[C++]=P}}C<z&&(l.length=C);m=$c(m,4,!0);m=$c(m,16,u);m=$c(m,8,t);H(l,m);p&&Object.freeze(l);l=m}m=!!(8&l)||k&&!g.length;if(f&&!m){Ud(l)&&
(g=Uc(g),l=$d(l,e),e=Td(a,e,b,g));f=g;for(m=0;m<f.length;m++)q=f[m],p=Pd(q),q!==p&&(f[m]=p);l=$c(l,8,!0);l=$c(l,16,!f.length);H(f,l)}Ud(l)||(f=l,k&&(l=$c(l,!g.length||16&l&&(!n||32&l)?2:2048,!0)),l!==f&&H(g,l),k&&Object.freeze(g));h&&Ud(l)&&(g=Uc(g),l=$d(l,e),H(g,l),Td(a,e,b,g));b=g;c=null!=d?Dd(d,c):new c;b.push(c);Yc(c.s)&2?Zc(b,8):Zc(b,16)}
function be(a,b){var c=void 0===c?"":c;a=Cd(Qd(a,b));return null!=a?a:c}
function ce(a,b){b=Xd(a,de,b);return Cd(Qd(a,b))}
function ee(a,b,c){if(null!=c){if("number"!==typeof c)throw pd("int32");if(!Number.isFinite(c))throw pd("int32");c|=0}Sd(a,b,c)}
function K(a,b,c){return Sd(a,b,Bd(c))}
function fe(a,b,c){if(null!=c){if(!Number.isFinite(c))throw pd("enum");c|=0}return Sd(a,b,c)}
;function L(a,b,c){this.s=I(a,b,c)}
L.prototype.toJSON=function(){if(jd)var a=ge(this,this.s,!1);else a=Kd(this.s,Ld,void 0,void 0,!1,!1),a=ge(this,a,!0);return a};
function he(a){jd=!0;try{return JSON.stringify(a.toJSON(),Gd)}finally{jd=!1}}
L.prototype.clone=function(){var a=this.s;return Nd(this,a,ad(a),!1)};
L.prototype.Tb=fd;L.prototype.toString=function(){return ge(this,this.s,!1).toString()};
function ge(a,b,c){var d=a.constructor.ma,e=ad(c?a.s:b),f=ed(e),g=!1;if(d&&kd){if(!c){b=Uc(b);var h;if(b.length&&id(h=b[b.length-1]))for(g=0;g<d.length;g++)if(d[g]>=f){Object.assign(b[b.length-1]={},h);break}g=!0}f=b;c=!c;h=ad(a.s);a=ed(h);h=+!!(h&512)-1;for(var k,l,n=0;n<d.length;n++)if(l=d[n],l<a){l+=h;var m=f[l];null==m?f[l]=c?md:bd():c&&m!==md&&Xc(m)}else k||(m=void 0,f.length&&id(m=f[f.length-1])?k=m:f.push(k={})),m=k[l],null==k[l]?k[l]=c?md:bd():c&&m!==md&&Xc(m)}k=b.length;if(!k)return b;var q;
if(id(f=b[k-1])){a:{var p=f;c={};a=!1;for(var t in p){h=p[t];if(Array.isArray(h)){n=h;if(!Mc&&ld(h,d,+t)||!Lc&&hd(h)&&0===h.size)h=null;h!=n&&(a=!0)}null!=h?c[t]=h:a=!0}if(a){for(var u in c){p=c;break a}p=null}}p!=f&&(q=!0);k--}for(e=+!!(e&512)-1;0<k;k--){t=k-1;f=b[t];if(!(null==f||!Mc&&ld(f,d,t-e)||!Lc&&hd(f)&&0===f.size))break;var z=!0}if(!q&&!z)return b;var C;g?C=b:C=Array.prototype.slice.call(b,0,k);b=C;g&&(b.length=k);p&&b.push(p);return b}
;var ie=window;wb("csi.gstatic.com");wb("googleads.g.doubleclick.net");wb("partner.googleadservices.com");wb("pubads.g.doubleclick.net");wb("securepubads.g.doubleclick.net");wb("tpc.googlesyndication.com");function je(a,b){this.width=a;this.height=b}
r=je.prototype;r.clone=function(){return new je(this.width,this.height)};
r.aspectRatio=function(){return this.width/this.height};
r.Nb=function(){return!(this.width*this.height)};
r.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
r.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
r.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};function ke(){var a=document;var b="IFRAME";"application/xhtml+xml"===a.contentType&&(b=b.toLowerCase());return a.createElement(b)}
function le(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;function me(a){var b=E("window.location.href");null==a&&(a='Unknown Error of type "null/undefined"');if("string"===typeof a)return{message:a,name:"Unknown error",lineNumber:"Not available",fileName:b,stack:"Not available"};var c=!1;try{var d=a.lineNumber||a.line||"Not available"}catch(g){d="Not available",c=!0}try{var e=a.fileName||a.filename||a.sourceURL||D.$googDebugFname||b}catch(g){e="Not available",c=!0}b=ne(a);if(!(!c&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name)){c=a.message;if(null==
c){if(a.constructor&&a.constructor instanceof Function){if(a.constructor.name)c=a.constructor.name;else if(c=a.constructor,oe[c])c=oe[c];else{c=String(c);if(!oe[c]){var f=/function\s+([^\(]+)/m.exec(c);oe[c]=f?f[1]:"[Anonymous]"}c=oe[c]}c='Unknown Error of type "'+c+'"'}else c="Unknown Error of unknown type";"function"===typeof a.toString&&Object.prototype.toString!==a.toString&&(c+=": "+a.toString())}return{message:c,name:a.name||"UnknownError",lineNumber:d,fileName:e,stack:b||"Not available"}}return{message:a.message,
name:a.name,lineNumber:a.lineNumber,fileName:a.fileName,stack:b}}
function ne(a,b){b||(b={});b[pe(a)]=!0;var c=a.stack||"";(a=a.cause)&&!b[pe(a)]&&(c+="\nCaused by: ",a.stack&&0==a.stack.indexOf(a.toString())||(c+="string"===typeof a?a:a.message+"\n"),c+=ne(a,b));return c}
function pe(a){var b="";"function"===typeof a.toString&&(b=""+a);return b+a.stack}
var oe={};/*

 SPDX-License-Identifier: Apache-2.0
*/
function qe(a){this.nd=a}
function re(a){return new qe(function(b){return b.substr(0,a.length+1).toLowerCase()===a+":"})}
var se=[re("data"),re("http"),re("https"),re("mailto"),re("ftp"),new qe(function(a){return/^[^:]*([/?#]|$)/.test(a)})];
function te(a,b){b=void 0===b?se:b;if(a instanceof Hb)return a;for(var c=0;c<b.length;++c){var d=b[c];if(d instanceof qe&&d.nd(a))return new Hb(a,Ib)}}
function ue(a){var b=void 0===b?se:b;return te(a,b)||Jb}
var ve="function"===typeof URL;function we(){throw Error("unknown trace type");}
;var xe={Sd:0,Xd:1,Qd:2,Rd:3,0:"FORMATTED_HTML_CONTENT",1:"HTML_FORMATTED_CONTENT",2:"EMBEDDED_INTERNAL_CONTENT",3:"EMBEDDED_TRUSTED_EXTERNAL_CONTENT"};function ye(a,b){b=Error.call(this,a+" cannot be used with intent "+xe[b]);this.message=b.message;"stack"in b&&(this.stack=b.stack);this.type=a;this.name="TypeCannotBeUsedWithIntentError"}
y(ye,Error);
function ze(a,b){a.removeAttribute("srcdoc");if(b instanceof xb)throw new ye("TrustedResourceUrl",3);var c="allow-same-origin allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox allow-storage-access-by-user-activation".split(" ");a.setAttribute("sandbox","");for(var d=0;d<c.length;d++)a.sandbox.supports&&!a.sandbox.supports(c[d])||a.sandbox.add(c[d]);if(b instanceof Hb)b instanceof Hb&&b.constructor===Hb?b=b.g:(Ka(b),b="type_error:SafeUrl");else{b:if(ve){try{var e=new URL(b)}catch(f){c="https:";
break b}c=e.protocol}else c:{c=document.createElement("a");try{c.href=b}catch(f){c=void 0;break c}c=c.protocol;c=":"===c||""===c?"https:":c}b="javascript:"!==c?b:void 0}void 0!==b&&(a.src=b)}
;function Ae(a){var b=Be;if(b)for(var c in b)Object.prototype.hasOwnProperty.call(b,c)&&a(b[c],c,b)}
function Ce(){var a=[];Ae(function(b){a.push(b)});
return a}
var Be={Ed:"allow-forms",Fd:"allow-modals",Gd:"allow-orientation-lock",Hd:"allow-pointer-lock",Id:"allow-popups",Jd:"allow-popups-to-escape-sandbox",Kd:"allow-presentation",Ld:"allow-same-origin",Md:"allow-scripts",Nd:"allow-top-navigation",Od:"allow-top-navigation-by-user-activation"},De=bb(function(){return Ce()});
function Ee(){var a=Fe(),b={};db(De(),function(c){a.sandbox&&a.sandbox.supports&&a.sandbox.supports(c)&&(b[c]=!0)});
return b}
function Fe(){var a=void 0===a?document:a;return a.createElement("iframe")}
;var Ge=(new Date).getTime();function He(a){if(!a)return"";if(/^about:(?:blank|srcdoc)$/.test(a))return window.origin||"";a.startsWith("blob:")&&(a=a.substring(5));a=a.split("#")[0].split("?")[0];a=a.toLowerCase();0==a.indexOf("//")&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");-1!=c&&(b=b.substring(0,c));c=a.substring(0,a.indexOf("://"));if(!c)throw Error("URI is missing protocol: "+a);if("http"!==c&&"https"!==c&&"chrome-extension"!==
c&&"moz-extension"!==c&&"file"!==c&&"android-app"!==c&&"chrome-search"!==c&&"chrome-untrusted"!==c&&"chrome"!==c&&"app"!==c&&"devtools"!==c)throw Error("Invalid URI scheme in origin: "+c);a="";var d=b.indexOf(":");if(-1!=d){var e=b.substring(d+1);b=b.substring(0,d);if("http"===c&&"80"!==e||"https"===c&&"443"!==e)a=":"+e}return c+"://"+b+a}
;function Ie(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;n=l=0}
function b(m){for(var q=g,p=0;64>p;p+=4)q[p/4]=m[p]<<24|m[p+1]<<16|m[p+2]<<8|m[p+3];for(p=16;80>p;p++)m=q[p-3]^q[p-8]^q[p-14]^q[p-16],q[p]=(m<<1|m>>>31)&4294967295;m=e[0];var t=e[1],u=e[2],z=e[3],C=e[4];for(p=0;80>p;p++){if(40>p)if(20>p){var P=z^t&(u^z);var U=1518500249}else P=t^u^z,U=1859775393;else 60>p?(P=t&u|z&(t|u),U=2400959708):(P=t^u^z,U=3395469782);P=((m<<5|m>>>27)&4294967295)+P+C+U+q[p]&4294967295;C=z;z=u;u=(t<<30|t>>>2)&4294967295;t=m;m=P}e[0]=e[0]+m&4294967295;e[1]=e[1]+t&4294967295;e[2]=
e[2]+u&4294967295;e[3]=e[3]+z&4294967295;e[4]=e[4]+C&4294967295}
function c(m,q){if("string"===typeof m){m=unescape(encodeURIComponent(m));for(var p=[],t=0,u=m.length;t<u;++t)p.push(m.charCodeAt(t));m=p}q||(q=m.length);p=0;if(0==l)for(;p+64<q;)b(m.slice(p,p+64)),p+=64,n+=64;for(;p<q;)if(f[l++]=m[p++],n++,64==l)for(l=0,b(f);p+64<q;)b(m.slice(p,p+64)),p+=64,n+=64}
function d(){var m=[],q=8*n;56>l?c(h,56-l):c(h,64-(l-56));for(var p=63;56<=p;p--)f[p]=q&255,q>>>=8;b(f);for(p=q=0;5>p;p++)for(var t=24;0<=t;t-=8)m[q++]=e[p]>>t&255;return m}
for(var e=[],f=[],g=[],h=[128],k=1;64>k;++k)h[k]=0;var l,n;a();return{reset:a,update:c,digest:d,Wc:function(){for(var m=d(),q="",p=0;p<m.length;p++)q+="0123456789ABCDEF".charAt(Math.floor(m[p]/16))+"0123456789ABCDEF".charAt(m[p]%16);return q}}}
;function Je(a,b,c){var d=String(D.location.href);return d&&a&&b?[b,Ke(He(d),a,c||null)].join(" "):null}
function Ke(a,b,c){var d=[],e=[];if(1==(Array.isArray(c)?2:1))return e=[b,a],db(d,function(h){e.push(h)}),Le(e.join(" "));
var f=[],g=[];db(c,function(h){g.push(h.key);f.push(h.value)});
c=Math.floor((new Date).getTime()/1E3);e=0==f.length?[c,b,a]:[f.join(":"),c,b,a];db(d,function(h){e.push(h)});
a=Le(e.join(" "));a=[c,a];0==g.length||a.push(g.join(""));return a.join("_")}
function Le(a){var b=Ie();b.update(a);return b.Wc().toLowerCase()}
;var Me={};function Ne(a){this.g=a||{cookie:""}}
r=Ne.prototype;r.isEnabled=function(){if(!D.navigator.cookieEnabled)return!1;if(!this.Nb())return!0;this.set("TESTCOOKIESENABLED","1",{Qb:60});if("1"!==this.get("TESTCOOKIESENABLED"))return!1;this.remove("TESTCOOKIESENABLED");return!0};
r.set=function(a,b,c){var d=!1;if("object"===typeof c){var e=c.ye;d=c.secure||!1;var f=c.domain||void 0;var g=c.path||void 0;var h=c.Qb}if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');void 0===h&&(h=-1);c=f?";domain="+f:"";g=g?";path="+g:"";d=d?";secure":"";h=0>h?"":0==h?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(Date.now()+1E3*h)).toUTCString();this.g.cookie=a+"="+b+c+g+h+d+(null!=e?";samesite="+
e:"")};
r.get=function(a,b){for(var c=a+"=",d=(this.g.cookie||"").split(";"),e=0,f;e<d.length;e++){f=zb(d[e]);if(0==f.lastIndexOf(c,0))return f.slice(c.length);if(f==a)return""}return b};
r.remove=function(a,b,c){var d=void 0!==this.get(a);this.set(a,"",{Qb:0,path:b,domain:c});return d};
r.Nb=function(){return!this.g.cookie};
r.clear=function(){for(var a=(this.g.cookie||"").split(";"),b=[],c=[],d,e,f=0;f<a.length;f++)e=zb(a[f]),d=e.indexOf("="),-1==d?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));for(a=b.length-1;0<=a;a--)this.remove(b[a])};
var Oe=new Ne("undefined"==typeof document?null:document);function Pe(a){return!!Me.FPA_SAMESITE_PHASE2_MOD||!(void 0===a||!a)}
function Qe(a,b,c,d){(a=D[a])||"undefined"===typeof document||(a=(new Ne(document)).get(b));return a?Je(a,c,d):null}
function Re(a){var b=void 0===b?!1:b;var c=He(String(D.location.href)),d=[];var e=b;e=void 0===e?!1:e;var f=D.__SAPISID||D.__APISID||D.__3PSAPISID||D.__OVERRIDE_SID;Pe(e)&&(f=f||D.__1PSAPISID);if(f)e=!0;else{if("undefined"!==typeof document){var g=new Ne(document);f=g.get("SAPISID")||g.get("APISID")||g.get("__Secure-3PAPISID")||g.get("SID")||g.get("OSID");Pe(e)&&(f=f||g.get("__Secure-1PAPISID"))}e=!!f}e&&(e=(c=0==c.indexOf("https:")||0==c.indexOf("chrome-extension:")||0==c.indexOf("moz-extension:"))?
D.__SAPISID:D.__APISID,e||"undefined"===typeof document||(e=new Ne(document),e=e.get(c?"SAPISID":"APISID")||e.get("__Secure-3PAPISID")),(e=e?Je(e,c?"SAPISIDHASH":"APISIDHASH",a):null)&&d.push(e),c&&Pe(b)&&((b=Qe("__1PSAPISID","__Secure-1PAPISID","SAPISID1PHASH",a))&&d.push(b),(a=Qe("__3PSAPISID","__Secure-3PAPISID","SAPISID3PHASH",a))&&d.push(a)));return 0==d.length?null:d.join(" ")}
;function Se(a){a&&"function"==typeof a.dispose&&a.dispose()}
;function We(){this.Aa=this.Aa;this.l=this.l}
We.prototype.Aa=!1;We.prototype.dispose=function(){this.Aa||(this.Aa=!0,this.sa())};
We.prototype.addOnDisposeCallback=function(a,b){this.Aa?void 0!==b?a.call(b):a():(this.l||(this.l=[]),this.l.push(void 0!==b?Va(a,b):a))};
We.prototype.sa=function(){if(this.l)for(;this.l.length;)this.l.shift()()};function Xe(a,b){this.type=a;this.g=this.target=b;this.defaultPrevented=this.i=!1}
Xe.prototype.stopPropagation=function(){this.i=!0};
Xe.prototype.preventDefault=function(){this.defaultPrevented=!0};var Ye=function(){if(!D.addEventListener||!Object.defineProperty)return!1;var a=!1,b=Object.defineProperty({},"passive",{get:function(){a=!0}});
try{var c=function(){};
D.addEventListener("test",c,b);D.removeEventListener("test",c,b)}catch(d){}return a}();function Ze(a,b){Xe.call(this,a?a.type:"");this.relatedTarget=this.g=this.target=null;this.button=this.screenY=this.screenX=this.clientY=this.clientX=0;this.key="";this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.altKey=this.ctrlKey=!1;this.state=null;this.pointerId=0;this.pointerType="";this.h=null;a&&this.init(a,b)}
Xa(Ze,Xe);var $e={2:"touch",3:"pen",4:"mouse"};
Ze.prototype.init=function(a,b){var c=this.type=a.type,d=a.changedTouches&&a.changedTouches.length?a.changedTouches[0]:null;this.target=a.target||a.srcElement;this.g=b;if(b=a.relatedTarget){if(nc){a:{try{jc(b.nodeName);var e=!0;break a}catch(f){}e=!1}e||(b=null)}}else"mouseover"==c?b=a.fromElement:"mouseout"==c&&(b=a.toElement);this.relatedTarget=b;d?(this.clientX=void 0!==d.clientX?d.clientX:d.pageX,this.clientY=void 0!==d.clientY?d.clientY:d.pageY,this.screenX=d.screenX||0,this.screenY=d.screenY||
0):(this.clientX=void 0!==a.clientX?a.clientX:a.pageX,this.clientY=void 0!==a.clientY?a.clientY:a.pageY,this.screenX=a.screenX||0,this.screenY=a.screenY||0);this.button=a.button;this.keyCode=a.keyCode||0;this.key=a.key||"";this.charCode=a.charCode||("keypress"==c?a.keyCode:0);this.ctrlKey=a.ctrlKey;this.altKey=a.altKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.pointerId=a.pointerId||0;this.pointerType="string"===typeof a.pointerType?a.pointerType:$e[a.pointerType]||"";this.state=a.state;
this.h=a;a.defaultPrevented&&Ze.na.preventDefault.call(this)};
Ze.prototype.stopPropagation=function(){Ze.na.stopPropagation.call(this);this.h.stopPropagation?this.h.stopPropagation():this.h.cancelBubble=!0};
Ze.prototype.preventDefault=function(){Ze.na.preventDefault.call(this);var a=this.h;a.preventDefault?a.preventDefault():a.returnValue=!1};var af="closure_listenable_"+(1E6*Math.random()|0);var bf=0;function cf(a,b,c,d,e){this.listener=a;this.proxy=null;this.src=b;this.type=c;this.capture=!!d;this.Ab=e;this.key=++bf;this.pb=this.vb=!1}
function df(a){a.pb=!0;a.listener=null;a.proxy=null;a.src=null;a.Ab=null}
;function ef(a){this.src=a;this.listeners={};this.g=0}
ef.prototype.add=function(a,b,c,d,e){var f=a.toString();a=this.listeners[f];a||(a=this.listeners[f]=[],this.g++);var g=ff(a,b,d,e);-1<g?(b=a[g],c||(b.vb=!1)):(b=new cf(b,this.src,f,!!d,e),b.vb=c,a.push(b));return b};
ef.prototype.remove=function(a,b,c,d){a=a.toString();if(!(a in this.listeners))return!1;var e=this.listeners[a];b=ff(e,b,c,d);return-1<b?(df(e[b]),Array.prototype.splice.call(e,b,1),0==e.length&&(delete this.listeners[a],this.g--),!0):!1};
function gf(a,b){var c=b.type;c in a.listeners&&eb(a.listeners[c],b)&&(df(b),0==a.listeners[c].length&&(delete a.listeners[c],a.g--))}
function ff(a,b,c,d){for(var e=0;e<a.length;++e){var f=a[e];if(!f.pb&&f.listener==b&&f.capture==!!c&&f.Ab==d)return e}return-1}
;var hf="closure_lm_"+(1E6*Math.random()|0),jf={},kf=0;function lf(a,b,c,d,e){if(d&&d.once)mf(a,b,c,d,e);else if(Array.isArray(b))for(var f=0;f<b.length;f++)lf(a,b[f],c,d,e);else c=nf(c),a&&a[af]?a.Ka(b,c,Oa(d)?!!d.capture:!!d,e):of(a,b,c,!1,d,e)}
function of(a,b,c,d,e,f){if(!b)throw Error("Invalid event type");var g=Oa(e)?!!e.capture:!!e,h=pf(a);h||(a[hf]=h=new ef(a));c=h.add(b,c,d,g,f);if(!c.proxy){d=qf();c.proxy=d;d.src=a;d.listener=c;if(a.addEventListener)Ye||(e=g),void 0===e&&(e=!1),a.addEventListener(b.toString(),d,e);else if(a.attachEvent)a.attachEvent(rf(b.toString()),d);else if(a.addListener&&a.removeListener)a.addListener(d);else throw Error("addEventListener and attachEvent are unavailable.");kf++}}
function qf(){function a(c){return b.call(a.src,a.listener,c)}
var b=sf;return a}
function mf(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)mf(a,b[f],c,d,e);else c=nf(c),a&&a[af]?a.g.add(String(b),c,!0,Oa(d)?!!d.capture:!!d,e):of(a,b,c,!0,d,e)}
function tf(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)tf(a,b[f],c,d,e);else(d=Oa(d)?!!d.capture:!!d,c=nf(c),a&&a[af])?a.g.remove(String(b),c,d,e):a&&(a=pf(a))&&(b=a.listeners[b.toString()],a=-1,b&&(a=ff(b,c,d,e)),(c=-1<a?b[a]:null)&&uf(c))}
function uf(a){if("number"!==typeof a&&a&&!a.pb){var b=a.src;if(b&&b[af])gf(b.g,a);else{var c=a.type,d=a.proxy;b.removeEventListener?b.removeEventListener(c,d,a.capture):b.detachEvent?b.detachEvent(rf(c),d):b.addListener&&b.removeListener&&b.removeListener(d);kf--;(c=pf(b))?(gf(c,a),0==c.g&&(c.src=null,b[hf]=null)):df(a)}}}
function rf(a){return a in jf?jf[a]:jf[a]="on"+a}
function sf(a,b){if(a.pb)a=!0;else{b=new Ze(b,this);var c=a.listener,d=a.Ab||a.src;a.vb&&uf(a);a=c.call(d,b)}return a}
function pf(a){a=a[hf];return a instanceof ef?a:null}
var vf="__closure_events_fn_"+(1E9*Math.random()>>>0);function nf(a){if("function"===typeof a)return a;a[vf]||(a[vf]=function(b){return a.handleEvent(b)});
return a[vf]}
;function wf(){We.call(this);this.g=new ef(this);this.O=this;this.G=null}
Xa(wf,We);wf.prototype[af]=!0;wf.prototype.addEventListener=function(a,b,c,d){lf(this,a,b,c,d)};
wf.prototype.removeEventListener=function(a,b,c,d){tf(this,a,b,c,d)};
function xf(a,b){var c=a.G;if(c){var d=[];for(var e=1;c;c=c.G)d.push(c),++e}a=a.O;c=b.type||b;"string"===typeof b?b=new Xe(b,a):b instanceof Xe?b.target=b.target||a:(e=b,b=new Xe(c,a),rb(b,e));e=!0;if(d)for(var f=d.length-1;!b.i&&0<=f;f--){var g=b.g=d[f];e=yf(g,c,!0,b)&&e}b.i||(g=b.g=a,e=yf(g,c,!0,b)&&e,b.i||(e=yf(g,c,!1,b)&&e));if(d)for(f=0;!b.i&&f<d.length;f++)g=b.g=d[f],e=yf(g,c,!1,b)&&e}
wf.prototype.sa=function(){wf.na.sa.call(this);if(this.g){var a=this.g,b=0,c;for(c in a.listeners){for(var d=a.listeners[c],e=0;e<d.length;e++)++b,df(d[e]);delete a.listeners[c];a.g--}}this.G=null};
wf.prototype.Ka=function(a,b,c,d){return this.g.add(String(a),b,!1,c,d)};
function yf(a,b,c,d){b=a.g.listeners[String(b)];if(!b)return!0;b=b.concat();for(var e=!0,f=0;f<b.length;++f){var g=b[f];if(g&&!g.pb&&g.capture==c){var h=g.listener,k=g.Ab||g.src;g.vb&&gf(a.g,g);e=!1!==h.call(k,d)&&e}}return e&&!d.defaultPrevented}
;function zf(a){wf.call(this);var b=this;this.K=this.i=0;this.ha=null!=a?a:{qa:function(e,f){return setTimeout(e,f)},
ba:function(e){clearTimeout(e)}};
var c,d;this.h=null!=(d=null==(c=window.navigator)?void 0:c.onLine)?d:!0;this.o=function(){return B(function(e){return A(e,Af(b),0)})};
window.addEventListener("offline",this.o);window.addEventListener("online",this.o);this.K||Bf(this)}
y(zf,wf);function Cf(){var a=Df;zf.g||(zf.g=new zf(a));return zf.g}
zf.prototype.dispose=function(){window.removeEventListener("offline",this.o);window.removeEventListener("online",this.o);this.ha.ba(this.K);delete zf.g};
zf.prototype.ca=function(){return this.h};
function Bf(a){a.K=a.ha.qa(function(){var b;return B(function(c){if(1==c.g)return a.h?(null==(b=window.navigator)?0:b.onLine)?c.B(3):A(c,Af(a),3):A(c,Af(a),3);Bf(a);c.g=0})},3E4)}
function Af(a,b){return a.u?a.u:a.u=new Promise(function(c){var d,e,f,g;return B(function(h){switch(h.g){case 1:return d=window.AbortController?new window.AbortController:void 0,f=null==(e=d)?void 0:e.signal,g=!1,va(h,2,3),d&&(a.i=a.ha.qa(function(){d.abort()},b||2E4)),A(h,fetch("/generate_204",{method:"HEAD",
signal:f}),5);case 5:g=!0;case 3:h.O=[h.i];h.o=0;h.u=0;a.u=void 0;a.i&&(a.ha.ba(a.i),a.i=0);g!==a.h&&(a.h=g,a.h?xf(a,"networkstatus-online"):xf(a,"networkstatus-offline"));c(g);xa(h);break;case 2:wa(h),g=!1,h.B(3)}})})}
;function Ef(){this.data=[];this.g=-1}
Ef.prototype.set=function(a,b){b=void 0===b?!0:b;0<=a&&52>a&&Number.isInteger(a)&&this.data[a]!==b&&(this.data[a]=b,this.g=-1)};
Ef.prototype.get=function(a){return!!this.data[a]};
function Ff(a){-1===a.g&&(a.g=a.data.reduce(function(b,c,d){return b+(c?Math.pow(2,d):0)},0));
return a.g}
;function Gf(a){this.s=I(a)}
y(Gf,L);function Hf(a){this.s=I(a)}
y(Hf,L);function If(a,b){return K(a,2,b)}
function Jf(a,b){return K(a,3,b)}
function Kf(a,b){return K(a,4,b)}
function Lf(a,b){return K(a,5,b)}
function Mf(a,b){return K(a,9,b)}
function Nf(a,b){var c=a.s,d=ad(c);od(d);if(null==b)Td(c,d,10);else{var e=Yc(b),f=e,g=!!(2&e)||!!(2048&e),h=g||Object.isFrozen(b),k;if(k=!h)k=!1;for(var l=!0,n=!0,m=0;m<b.length;m++){var q=b[m];Dd(q,Gf);g||(q=!!(Yc(q.s)&2),l&&(l=!q),n&&(n=q))}g||(e=$c(e,5,!0),e=$c(e,8,l),e=$c(e,16,n));if(k||h&&e!==f)b=Uc(b),f=0,e=$d(e,d);e!==f&&H(b,e);Td(c,d,10,b)}return a}
function Of(a,b){return Sd(a,11,null==b?b:rd(b))}
function Pf(a,b){return K(a,1,b)}
function Qf(a,b){return Sd(a,7,null==b?b:rd(b))}
Hf.ma=[10,6];var Rf="platform platformVersion architecture model uaFullVersion bitness fullVersionList wow64".split(" ");function Sf(a){var b;return null!=(b=a.google_tag_data)?b:a.google_tag_data={}}
function Tf(a){var b,c;return"function"===typeof(null==(b=a.navigator)?void 0:null==(c=b.userAgentData)?void 0:c.getHighEntropyValues)}
function Uf(){var a=window;if(!Tf(a))return null;var b=Sf(a);if(b.uach_promise)return b.uach_promise;a=a.navigator.userAgentData.getHighEntropyValues(Rf).then(function(c){null!=b.uach||(b.uach=c);return c});
return b.uach_promise=a}
function Vf(a){var b;return Of(Nf(Lf(If(Pf(Kf(Qf(Mf(Jf(new Hf,a.architecture||""),a.bitness||""),a.mobile||!1),a.model||""),a.platform||""),a.platformVersion||""),a.uaFullVersion||""),(null==(b=a.fullVersionList)?void 0:b.map(function(c){var d=new Gf;d=K(d,1,c.brand);return K(d,2,c.version)}))||[]),a.wow64||!1)}
function Wf(){var a,b;return null!=(b=null==(a=Uf())?void 0:a.then(function(c){return Vf(c)}))?b:null}
;function Xf(a,b){this.i=a;this.l=b;this.h=0;this.g=null}
Xf.prototype.get=function(){if(0<this.h){this.h--;var a=this.g;this.g=a.next;a.next=null}else a=this.i();return a};
function Yf(a,b){a.l(b);100>a.h&&(a.h++,b.next=a.g,a.g=b)}
;var Zf;function $f(){var a=D.MessageChannel;"undefined"===typeof a&&"undefined"!==typeof window&&window.postMessage&&window.addEventListener&&!G("Presto")&&(a=function(){var e=ke();e.style.display="none";document.documentElement.appendChild(e);var f=e.contentWindow;e=f.document;e.open();e.close();var g="callImmediate"+Math.random(),h="file:"==f.location.protocol?"*":f.location.protocol+"//"+f.location.host;e=Va(function(k){if(("*"==h||k.origin==h)&&k.data==g)this.port1.onmessage()},this);
f.addEventListener("message",e,!1);this.port1={};this.port2={postMessage:function(){f.postMessage(g,h)}}});
if("undefined"!==typeof a&&!Rb()){var b=new a,c={},d=c;b.port1.onmessage=function(){if(void 0!==c.next){c=c.next;var e=c.jc;c.jc=null;e()}};
return function(e){d.next={jc:e};d=d.next;b.port2.postMessage(0)}}return function(e){D.setTimeout(e,0)}}
;function ag(){this.h=this.g=null}
ag.prototype.add=function(a,b){var c=bg.get();c.set(a,b);this.h?this.h.next=c:this.g=c;this.h=c};
ag.prototype.remove=function(){var a=null;this.g&&(a=this.g,this.g=this.g.next,this.g||(this.h=null),a.next=null);return a};
var bg=new Xf(function(){return new cg},function(a){return a.reset()});
function cg(){this.next=this.scope=this.g=null}
cg.prototype.set=function(a,b){this.g=a;this.scope=b;this.next=null};
cg.prototype.reset=function(){this.next=this.scope=this.g=null};var dg,eg=!1,fg=new ag;function gg(a,b){dg||hg();eg||(dg(),eg=!0);fg.add(a,b)}
function hg(){if(D.Promise&&D.Promise.resolve){var a=D.Promise.resolve(void 0);dg=function(){a.then(ig)}}else dg=function(){var b=ig;
"function"!==typeof D.setImmediate||D.Window&&D.Window.prototype&&(Qb()||!G("Edge"))&&D.Window.prototype.setImmediate==D.setImmediate?(Zf||(Zf=$f()),Zf(b)):D.setImmediate(b)}}
function ig(){for(var a;a=fg.remove();){try{a.g.call(a.scope)}catch(b){hc(b)}Yf(bg,a)}eg=!1}
;function jg(a,b){this.g=a[D.Symbol.iterator]();this.h=b}
jg.prototype[Symbol.iterator]=function(){return this};
jg.prototype.next=function(){var a=this.g.next();return{value:a.done?void 0:this.h.call(void 0,a.value),done:a.done}};
function kg(a,b){return new jg(a,b)}
;function lg(){this.blockSize=-1}
;function mg(){this.blockSize=-1;this.blockSize=64;this.g=[];this.o=[];this.u=[];this.i=[];this.i[0]=128;for(var a=1;a<this.blockSize;++a)this.i[a]=0;this.l=this.h=0;this.reset()}
Xa(mg,lg);mg.prototype.reset=function(){this.g[0]=1732584193;this.g[1]=4023233417;this.g[2]=2562383102;this.g[3]=271733878;this.g[4]=3285377520;this.l=this.h=0};
function ng(a,b,c){c||(c=0);var d=a.u;if("string"===typeof b)for(var e=0;16>e;e++)d[e]=b.charCodeAt(c)<<24|b.charCodeAt(c+1)<<16|b.charCodeAt(c+2)<<8|b.charCodeAt(c+3),c+=4;else for(e=0;16>e;e++)d[e]=b[c]<<24|b[c+1]<<16|b[c+2]<<8|b[c+3],c+=4;for(e=16;80>e;e++){var f=d[e-3]^d[e-8]^d[e-14]^d[e-16];d[e]=(f<<1|f>>>31)&4294967295}b=a.g[0];c=a.g[1];var g=a.g[2],h=a.g[3],k=a.g[4];for(e=0;80>e;e++){if(40>e)if(20>e){f=h^c&(g^h);var l=1518500249}else f=c^g^h,l=1859775393;else 60>e?(f=c&g|h&(c|g),l=2400959708):
(f=c^g^h,l=3395469782);f=(b<<5|b>>>27)+f+k+l+d[e]&4294967295;k=h;h=g;g=(c<<30|c>>>2)&4294967295;c=b;b=f}a.g[0]=a.g[0]+b&4294967295;a.g[1]=a.g[1]+c&4294967295;a.g[2]=a.g[2]+g&4294967295;a.g[3]=a.g[3]+h&4294967295;a.g[4]=a.g[4]+k&4294967295}
mg.prototype.update=function(a,b){if(null!=a){void 0===b&&(b=a.length);for(var c=b-this.blockSize,d=0,e=this.o,f=this.h;d<b;){if(0==f)for(;d<=c;)ng(this,a,d),d+=this.blockSize;if("string"===typeof a)for(;d<b;){if(e[f]=a.charCodeAt(d),++f,++d,f==this.blockSize){ng(this,e);f=0;break}}else for(;d<b;)if(e[f]=a[d],++f,++d,f==this.blockSize){ng(this,e);f=0;break}}this.h=f;this.l+=b}};
mg.prototype.digest=function(){var a=[],b=8*this.l;56>this.h?this.update(this.i,56-this.h):this.update(this.i,this.blockSize-(this.h-56));for(var c=this.blockSize-1;56<=c;c--)this.o[c]=b&255,b/=256;ng(this,this.o);for(c=b=0;5>c;c++)for(var d=24;0<=d;d-=8)a[b]=this.g[c]>>d&255,++b;return a};function og(){}
og.prototype.next=function(){return pg};
var pg={done:!0,value:void 0};function qg(a){return{value:a,done:!1}}
og.prototype.ia=function(){return this};function rg(a){if(a instanceof sg||a instanceof tg||a instanceof ug)return a;if("function"==typeof a.next)return new sg(function(){return a});
if("function"==typeof a[Symbol.iterator])return new sg(function(){return a[Symbol.iterator]()});
if("function"==typeof a.ia)return new sg(function(){return a.ia()});
throw Error("Not an iterator or iterable.");}
function sg(a){this.h=a}
sg.prototype.ia=function(){return new tg(this.h())};
sg.prototype[Symbol.iterator]=function(){return new ug(this.h())};
sg.prototype.g=function(){return new ug(this.h())};
function tg(a){this.h=a}
y(tg,og);tg.prototype.next=function(){return this.h.next()};
tg.prototype[Symbol.iterator]=function(){return new ug(this.h)};
tg.prototype.g=function(){return new ug(this.h)};
function ug(a){sg.call(this,function(){return a});
this.i=a}
y(ug,sg);ug.prototype.next=function(){return this.i.next()};function vg(a,b){this.h={};this.g=[];this.i=this.size=0;var c=arguments.length;if(1<c){if(c%2)throw Error("Uneven number of arguments");for(var d=0;d<c;d+=2)this.set(arguments[d],arguments[d+1])}else if(a)if(a instanceof vg)for(c=wg(a),d=0;d<c.length;d++)this.set(c[d],a.get(c[d]));else for(d in a)this.set(d,a[d])}
function wg(a){xg(a);return a.g.concat()}
r=vg.prototype;r.has=function(a){return yg(this.h,a)};
r.equals=function(a,b){if(this===a)return!0;if(this.size!=a.size)return!1;b=b||zg;xg(this);for(var c,d=0;c=this.g[d];d++)if(!b(this.get(c),a.get(c)))return!1;return!0};
function zg(a,b){return a===b}
r.Nb=function(){return 0==this.size};
r.clear=function(){this.h={};this.i=this.size=this.g.length=0};
r.remove=function(a){return this.delete(a)};
r.delete=function(a){return yg(this.h,a)?(delete this.h[a],--this.size,this.i++,this.g.length>2*this.size&&xg(this),!0):!1};
function xg(a){if(a.size!=a.g.length){for(var b=0,c=0;b<a.g.length;){var d=a.g[b];yg(a.h,d)&&(a.g[c++]=d);b++}a.g.length=c}if(a.size!=a.g.length){var e={};for(c=b=0;b<a.g.length;)d=a.g[b],yg(e,d)||(a.g[c++]=d,e[d]=1),b++;a.g.length=c}}
r.get=function(a,b){return yg(this.h,a)?this.h[a]:b};
r.set=function(a,b){yg(this.h,a)||(this.size+=1,this.g.push(a),this.i++);this.h[a]=b};
r.forEach=function(a,b){for(var c=wg(this),d=0;d<c.length;d++){var e=c[d],f=this.get(e);a.call(b,f,e,this)}};
r.clone=function(){return new vg(this)};
r.keys=function(){return rg(this.ia(!0)).g()};
r.values=function(){return rg(this.ia(!1)).g()};
r.entries=function(){var a=this;return kg(this.keys(),function(b){return[b,a.get(b)]})};
r.ia=function(a){xg(this);var b=0,c=this.i,d=this,e=new og;e.next=function(){if(c!=d.i)throw Error("The map has changed since the iterator was created");if(b>=d.g.length)return pg;var f=d.g[b++];return qg(a?f:d.h[f])};
return e};
function yg(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
;var Ag=D.JSON.stringify;function Bg(){var a=this;this.promise=new Promise(function(b,c){a.resolve=b;a.reject=c})}
;function Cg(a){this.g=0;this.G=void 0;this.l=this.h=this.i=null;this.o=this.u=!1;if(a!=ab)try{var b=this;a.call(void 0,function(c){Dg(b,2,c)},function(c){Dg(b,3,c)})}catch(c){Dg(this,3,c)}}
function Eg(){this.next=this.context=this.h=this.i=this.g=null;this.l=!1}
Eg.prototype.reset=function(){this.context=this.h=this.i=this.g=null;this.l=!1};
var Fg=new Xf(function(){return new Eg},function(a){a.reset()});
function Gg(a,b,c){var d=Fg.get();d.i=a;d.h=b;d.context=c;return d}
Cg.prototype.then=function(a,b,c){return Hg(this,"function"===typeof a?a:null,"function"===typeof b?b:null,c)};
Cg.prototype.$goog_Thenable=!0;Cg.prototype.cancel=function(a){if(0==this.g){var b=new Ig(a);gg(function(){Jg(this,b)},this)}};
function Jg(a,b){if(0==a.g)if(a.i){var c=a.i;if(c.h){for(var d=0,e=null,f=null,g=c.h;g&&(g.l||(d++,g.g==a&&(e=g),!(e&&1<d)));g=g.next)e||(f=g);e&&(0==c.g&&1==d?Jg(c,b):(f?(d=f,d.next==c.l&&(c.l=d),d.next=d.next.next):Kg(c),Lg(c,e,3,b)))}a.i=null}else Dg(a,3,b)}
function Mg(a,b){a.h||2!=a.g&&3!=a.g||Ng(a);a.l?a.l.next=b:a.h=b;a.l=b}
function Hg(a,b,c,d){var e=Gg(null,null,null);e.g=new Cg(function(f,g){e.i=b?function(h){try{var k=b.call(d,h);f(k)}catch(l){g(l)}}:f;
e.h=c?function(h){try{var k=c.call(d,h);void 0===k&&h instanceof Ig?g(h):f(k)}catch(l){g(l)}}:g});
e.g.i=a;Mg(a,e);return e.g}
Cg.prototype.O=function(a){this.g=0;Dg(this,2,a)};
Cg.prototype.oa=function(a){this.g=0;Dg(this,3,a)};
function Dg(a,b,c){if(0==a.g){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.g=1;a:{var d=c,e=a.O,f=a.oa;if(d instanceof Cg){Mg(d,Gg(e||ab,f||null,a));var g=!0}else{if(d)try{var h=!!d.$goog_Thenable}catch(l){h=!1}else h=!1;if(h)d.then(e,f,a),g=!0;else{if(Oa(d))try{var k=d.then;if("function"===typeof k){Og(d,k,e,f,a);g=!0;break a}}catch(l){f.call(a,l);g=!0;break a}g=!1}}}g||(a.G=c,a.g=b,a.i=null,Ng(a),3!=b||c instanceof Ig||Pg(a,c))}}
function Og(a,b,c,d,e){function f(k){h||(h=!0,d.call(e,k))}
function g(k){h||(h=!0,c.call(e,k))}
var h=!1;try{b.call(a,g,f)}catch(k){f(k)}}
function Ng(a){a.u||(a.u=!0,gg(a.K,a))}
function Kg(a){var b=null;a.h&&(b=a.h,a.h=b.next,b.next=null);a.h||(a.l=null);return b}
Cg.prototype.K=function(){for(var a;a=Kg(this);)Lg(this,a,this.g,this.G);this.u=!1};
function Lg(a,b,c,d){if(3==c&&b.h&&!b.l)for(;a&&a.o;a=a.i)a.o=!1;if(b.g)b.g.i=null,Qg(b,c,d);else try{b.l?b.i.call(b.context):Qg(b,c,d)}catch(e){Rg.call(null,e)}Yf(Fg,b)}
function Qg(a,b,c){2==b?a.i.call(a.context,c):a.h&&a.h.call(a.context,c)}
function Pg(a,b){a.o=!0;gg(function(){a.o&&Rg.call(null,b)})}
var Rg=hc;function Ig(a){Za.call(this,a)}
Xa(Ig,Za);Ig.prototype.name="cancel";function M(a){We.call(this);this.u=1;this.i=[];this.o=0;this.g=[];this.h={};this.G=!!a}
Xa(M,We);r=M.prototype;r.subscribe=function(a,b,c){var d=this.h[a];d||(d=this.h[a]=[]);var e=this.u;this.g[e]=a;this.g[e+1]=b;this.g[e+2]=c;this.u=e+3;d.push(e);return e};
function Sg(a,b,c){var d=Tg;if(a=d.h[a]){var e=d.g;(a=a.find(function(f){return e[f+1]==b&&e[f+2]==c}))&&d.rb(a)}}
r.rb=function(a){var b=this.g[a];if(b){var c=this.h[b];0!=this.o?(this.i.push(a),this.g[a+1]=function(){}):(c&&eb(c,a),delete this.g[a],delete this.g[a+1],delete this.g[a+2])}return!!b};
r.ab=function(a,b){var c=this.h[a];if(c){for(var d=Array(arguments.length-1),e=1,f=arguments.length;e<f;e++)d[e-1]=arguments[e];if(this.G)for(e=0;e<c.length;e++){var g=c[e];Ug(this.g[g+1],this.g[g+2],d)}else{this.o++;try{for(e=0,f=c.length;e<f&&!this.Aa;e++)g=c[e],this.g[g+1].apply(this.g[g+2],d)}finally{if(this.o--,0<this.i.length&&0==this.o)for(;c=this.i.pop();)this.rb(c)}}return 0!=e}return!1};
function Ug(a,b,c){gg(function(){a.apply(b,c)})}
r.clear=function(a){if(a){var b=this.h[a];b&&(b.forEach(this.rb,this),delete this.h[a])}else this.g.length=0,this.h={}};
r.sa=function(){M.na.sa.call(this);this.clear();this.i.length=0};function Vg(a){this.g=a}
Vg.prototype.set=function(a,b){void 0===b?this.g.remove(a):this.g.set(a,Ag(b))};
Vg.prototype.get=function(a){try{var b=this.g.get(a)}catch(c){return}if(null!==b)try{return JSON.parse(b)}catch(c){throw"Storage: Invalid value was encountered";}};
Vg.prototype.remove=function(a){this.g.remove(a)};function Wg(a){this.g=a}
Xa(Wg,Vg);function Xg(a){this.data=a}
function Yg(a){return void 0===a||a instanceof Xg?a:new Xg(a)}
Wg.prototype.set=function(a,b){Wg.na.set.call(this,a,Yg(b))};
Wg.prototype.h=function(a){a=Wg.na.get.call(this,a);if(void 0===a||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
Wg.prototype.get=function(a){if(a=this.h(a)){if(a=a.data,void 0===a)throw"Storage: Invalid value was encountered";}else a=void 0;return a};function Zg(a){this.g=a}
Xa(Zg,Wg);Zg.prototype.set=function(a,b,c){if(b=Yg(b)){if(c){if(c<Wa()){Zg.prototype.remove.call(this,a);return}b.expiration=c}b.creation=Wa()}Zg.na.set.call(this,a,b)};
Zg.prototype.h=function(a){var b=Zg.na.h.call(this,a);if(b){var c=b.creation,d=b.expiration;if(d&&d<Wa()||c&&c>Wa())Zg.prototype.remove.call(this,a);else return b}};function $g(){}
;function ah(){}
Xa(ah,$g);ah.prototype[Symbol.iterator]=function(){return rg(this.ia(!0)).g()};
ah.prototype.clear=function(){var a=Array.from(this);a=w(a);for(var b=a.next();!b.done;b=a.next())this.remove(b.value)};function bh(a){this.g=a;this.h=null}
Xa(bh,ah);r=bh.prototype;r.set=function(a,b){ch(this);try{this.g.setItem(a,b)}catch(c){if(0==this.g.length)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
r.get=function(a){ch(this);a=this.g.getItem(a);if("string"!==typeof a&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
r.remove=function(a){ch(this);this.g.removeItem(a)};
r.ia=function(a){ch(this);var b=0,c=this.g,d=new og;d.next=function(){if(b>=c.length)return pg;var e=c.key(b++);if(a)return qg(e);e=c.getItem(e);if("string"!==typeof e)throw"Storage mechanism: Invalid value was encountered";return qg(e)};
return d};
r.clear=function(){ch(this);this.g.clear()};
r.key=function(a){ch(this);return this.g.key(a)};
function ch(a){if(null==a.g)throw Error("Storage mechanism: Storage unavailable");var b;(null!=(b=a.h)?b:a.h=dh(a.g))||hc(Error("Storage mechanism: Storage unavailable"))}
function dh(a){if(!a)return!1;try{return a.setItem("__sak","1"),a.removeItem("__sak"),!0}catch(b){return b instanceof DOMException&&("QuotaExceededError"===b.name||22===b.code||1014===b.code||"NS_ERROR_DOM_QUOTA_REACHED"===b.name)&&a&&0!==a.length}}
;function eh(){var a=null;try{a=D.localStorage||null}catch(b){}bh.call(this,a)}
Xa(eh,bh);function fh(a,b){this.h=a;this.g=null;var c;if(c=lc)c=!(9<=Number(xc));if(c){gh||(gh=new vg);this.g=gh.get(a);this.g||(b?this.g=document.getElementById(b):(this.g=document.createElement("userdata"),this.g.addBehavior("#default#userData"),document.body.appendChild(this.g)),gh.set(a,this.g));try{this.g.load(this.h)}catch(d){this.g=null}}}
Xa(fh,ah);var hh={".":".2E","!":".21","~":".7E","*":".2A","'":".27","(":".28",")":".29","%":"."},gh=null;function ih(a){return"_"+encodeURIComponent(a).replace(/[.!~*'()%]/g,function(b){return hh[b]})}
r=fh.prototype;r.set=function(a,b){this.g.setAttribute(ih(a),b);jh(this)};
r.get=function(a){a=this.g.getAttribute(ih(a));if("string"!==typeof a&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
r.remove=function(a){this.g.removeAttribute(ih(a));jh(this)};
r.ia=function(a){var b=0,c=this.g.XMLDocument.documentElement.attributes,d=new og;d.next=function(){if(b>=c.length)return pg;var e=c[b++];if(a)return qg(decodeURIComponent(e.nodeName.replace(/\./g,"%")).slice(1));e=e.nodeValue;if("string"!==typeof e)throw"Storage mechanism: Invalid value was encountered";return qg(e)};
return d};
r.clear=function(){for(var a=this.g.XMLDocument.documentElement,b=a.attributes.length;0<b;b--)a.removeAttribute(a.attributes[b-1].nodeName);jh(this)};
function jh(a){try{a.g.save(a.h)}catch(b){throw"Storage mechanism: Quota exceeded";}}
;function kh(a,b){this.h=a;this.g=b+"::"}
Xa(kh,ah);kh.prototype.set=function(a,b){this.h.set(this.g+a,b)};
kh.prototype.get=function(a){return this.h.get(this.g+a)};
kh.prototype.remove=function(a){this.h.remove(this.g+a)};
kh.prototype.ia=function(a){var b=this.h[Symbol.iterator](),c=this,d=new og;d.next=function(){var e=b.next();if(e.done)return e;for(e=e.value;e.slice(0,c.g.length)!=c.g;){e=b.next();if(e.done)return e;e=e.value}return qg(a?e.slice(c.g.length):c.h.get(e))};
return d};/*

 (The MIT License)

 Copyright (C) 2014 by Vitaly Puzrin

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.

 -----------------------------------------------------------------------------
 Ported from zlib, which is under the following license
 https://github.com/madler/zlib/blob/master/zlib.h

 zlib.h -- interface of the 'zlib' general purpose compression library
   version 1.2.8, April 28th, 2013
   Copyright (C) 1995-2013 Jean-loup Gailly and Mark Adler
   This software is provided 'as-is', without any express or implied
   warranty.  In no event will the authors be held liable for any damages
   arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:
   1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
   Jean-loup Gailly        Mark Adler
   jloup@gzip.org          madler@alumni.caltech.edu
   The data format used by the zlib library is described by RFCs (Request for
   Comments) 1950 to 1952 in the files http://tools.ietf.org/html/rfc1950
   (zlib format), rfc1951 (deflate format) and rfc1952 (gzip format).
*/
var N={},lh="undefined"!==typeof Uint8Array&&"undefined"!==typeof Uint16Array&&"undefined"!==typeof Int32Array;N.assign=function(a){for(var b=Array.prototype.slice.call(arguments,1);b.length;){var c=b.shift();if(c){if("object"!==typeof c)throw new TypeError(c+"must be non-object");for(var d in c)Object.prototype.hasOwnProperty.call(c,d)&&(a[d]=c[d])}}return a};
N.ac=function(a,b){if(a.length===b)return a;if(a.subarray)return a.subarray(0,b);a.length=b;return a};
var mh={Qa:function(a,b,c,d,e){if(b.subarray&&a.subarray)a.set(b.subarray(c,c+d),e);else for(var f=0;f<d;f++)a[e+f]=b[c+f]},
nc:function(a){var b,c;var d=c=0;for(b=a.length;d<b;d++)c+=a[d].length;var e=new Uint8Array(c);d=c=0;for(b=a.length;d<b;d++){var f=a[d];e.set(f,c);c+=f.length}return e}},nh={Qa:function(a,b,c,d,e){for(var f=0;f<d;f++)a[e+f]=b[c+f]},
nc:function(a){return[].concat.apply([],a)}};
N.xd=function(){lh?(N.Pa=Uint8Array,N.pa=Uint16Array,N.Lc=Int32Array,N.assign(N,mh)):(N.Pa=Array,N.pa=Array,N.Lc=Array,N.assign(N,nh))};
N.xd();var oh=!0;try{new Uint8Array(1)}catch(a){oh=!1}
function ph(a){var b,c,d=a.length,e=0;for(b=0;b<d;b++){var f=a.charCodeAt(b);if(55296===(f&64512)&&b+1<d){var g=a.charCodeAt(b+1);56320===(g&64512)&&(f=65536+(f-55296<<10)+(g-56320),b++)}e+=128>f?1:2048>f?2:65536>f?3:4}var h=new N.Pa(e);for(b=c=0;c<e;b++)f=a.charCodeAt(b),55296===(f&64512)&&b+1<d&&(g=a.charCodeAt(b+1),56320===(g&64512)&&(f=65536+(f-55296<<10)+(g-56320),b++)),128>f?h[c++]=f:(2048>f?h[c++]=192|f>>>6:(65536>f?h[c++]=224|f>>>12:(h[c++]=240|f>>>18,h[c++]=128|f>>>12&63),h[c++]=128|f>>>
6&63),h[c++]=128|f&63);return h}
;var qh={};qh=function(a,b,c,d){var e=a&65535|0;a=a>>>16&65535|0;for(var f;0!==c;){f=2E3<c?2E3:c;c-=f;do e=e+b[d++]|0,a=a+e|0;while(--f);e%=65521;a%=65521}return e|a<<16|0};for(var rh={},sh,th=[],uh=0;256>uh;uh++){sh=uh;for(var vh=0;8>vh;vh++)sh=sh&1?3988292384^sh>>>1:sh>>>1;th[uh]=sh}rh=function(a,b,c,d){c=d+c;for(a^=-1;d<c;d++)a=a>>>8^th[(a^b[d])&255];return a^-1};var wh={};wh={2:"need dictionary",1:"stream end",0:"","-1":"file error","-2":"stream error","-3":"data error","-4":"insufficient memory","-5":"buffer error","-6":"incompatible version"};function xh(a){for(var b=a.length;0<=--b;)a[b]=0}
var yh=[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0],zh=[0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13],Ah=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7],Bh=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15],Ch=Array(576);xh(Ch);var Dh=Array(60);xh(Dh);var Eh=Array(512);xh(Eh);var Fh=Array(256);xh(Fh);var Gh=Array(29);xh(Gh);var Hh=Array(30);xh(Hh);function Ih(a,b,c,d,e){this.Hc=a;this.ad=b;this.Zc=c;this.Xc=d;this.rd=e;this.qc=a&&a.length}
var Jh,Kh,Lh;function Mh(a,b){this.lc=a;this.Xa=0;this.Da=b}
function Nh(a,b){a.L[a.pending++]=b&255;a.L[a.pending++]=b>>>8&255}
function Oh(a,b,c){a.P>16-c?(a.V|=b<<a.P&65535,Nh(a,a.V),a.V=b>>16-a.P,a.P+=c-16):(a.V|=b<<a.P&65535,a.P+=c)}
function Ph(a,b,c){Oh(a,c[2*b],c[2*b+1])}
function Qh(a,b){var c=0;do c|=a&1,a>>>=1,c<<=1;while(0<--b);return c>>>1}
function oi(a,b,c){var d=Array(16),e=0,f;for(f=1;15>=f;f++)d[f]=e=e+c[f-1]<<1;for(c=0;c<=b;c++)e=a[2*c+1],0!==e&&(a[2*c]=Qh(d[e]++,e))}
function pi(a){var b;for(b=0;286>b;b++)a.X[2*b]=0;for(b=0;30>b;b++)a.Ia[2*b]=0;for(b=0;19>b;b++)a.S[2*b]=0;a.X[512]=1;a.xa=a.bb=0;a.da=a.matches=0}
function qi(a){8<a.P?Nh(a,a.V):0<a.P&&(a.L[a.pending++]=a.V);a.V=0;a.P=0}
function ri(a,b,c){qi(a);Nh(a,c);Nh(a,~c);N.Qa(a.L,a.window,b,c,a.pending);a.pending+=c}
function si(a,b,c,d){var e=2*b,f=2*c;return a[e]<a[f]||a[e]===a[f]&&d[b]<=d[c]}
function ti(a,b,c){for(var d=a.M[c],e=c<<1;e<=a.wa;){e<a.wa&&si(b,a.M[e+1],a.M[e],a.depth)&&e++;if(si(b,d,a.M[e],a.depth))break;a.M[c]=a.M[e];c=e;e<<=1}a.M[c]=d}
function ui(a,b,c){var d=0;if(0!==a.da){do{var e=a.L[a.jb+2*d]<<8|a.L[a.jb+2*d+1];var f=a.L[a.Pb+d];d++;if(0===e)Ph(a,f,b);else{var g=Fh[f];Ph(a,g+256+1,b);var h=yh[g];0!==h&&(f-=Gh[g],Oh(a,f,h));e--;g=256>e?Eh[e]:Eh[256+(e>>>7)];Ph(a,g,c);h=zh[g];0!==h&&(e-=Hh[g],Oh(a,e,h))}}while(d<a.da)}Ph(a,256,b)}
function vi(a,b){var c=b.lc,d=b.Da.Hc,e=b.Da.qc,f=b.Da.Xc,g,h=-1;a.wa=0;a.Ta=573;for(g=0;g<f;g++)0!==c[2*g]?(a.M[++a.wa]=h=g,a.depth[g]=0):c[2*g+1]=0;for(;2>a.wa;){var k=a.M[++a.wa]=2>h?++h:0;c[2*k]=1;a.depth[k]=0;a.xa--;e&&(a.bb-=d[2*k+1])}b.Xa=h;for(g=a.wa>>1;1<=g;g--)ti(a,c,g);k=f;do g=a.M[1],a.M[1]=a.M[a.wa--],ti(a,c,1),d=a.M[1],a.M[--a.Ta]=g,a.M[--a.Ta]=d,c[2*k]=c[2*g]+c[2*d],a.depth[k]=(a.depth[g]>=a.depth[d]?a.depth[g]:a.depth[d])+1,c[2*g+1]=c[2*d+1]=k,a.M[1]=k++,ti(a,c,1);while(2<=a.wa);a.M[--a.Ta]=
a.M[1];g=b.lc;k=b.Xa;d=b.Da.Hc;e=b.Da.qc;f=b.Da.ad;var l=b.Da.Zc,n=b.Da.rd,m,q=0;for(m=0;15>=m;m++)a.ra[m]=0;g[2*a.M[a.Ta]+1]=0;for(b=a.Ta+1;573>b;b++){var p=a.M[b];m=g[2*g[2*p+1]+1]+1;m>n&&(m=n,q++);g[2*p+1]=m;if(!(p>k)){a.ra[m]++;var t=0;p>=l&&(t=f[p-l]);var u=g[2*p];a.xa+=u*(m+t);e&&(a.bb+=u*(d[2*p+1]+t))}}if(0!==q){do{for(m=n-1;0===a.ra[m];)m--;a.ra[m]--;a.ra[m+1]+=2;a.ra[n]--;q-=2}while(0<q);for(m=n;0!==m;m--)for(p=a.ra[m];0!==p;)d=a.M[--b],d>k||(g[2*d+1]!==m&&(a.xa+=(m-g[2*d+1])*g[2*d],g[2*
d+1]=m),p--)}oi(c,h,a.ra)}
function wi(a,b,c){var d,e=-1,f=b[1],g=0,h=7,k=4;0===f&&(h=138,k=3);b[2*(c+1)+1]=65535;for(d=0;d<=c;d++){var l=f;f=b[2*(d+1)+1];++g<h&&l===f||(g<k?a.S[2*l]+=g:0!==l?(l!==e&&a.S[2*l]++,a.S[32]++):10>=g?a.S[34]++:a.S[36]++,g=0,e=l,0===f?(h=138,k=3):l===f?(h=6,k=3):(h=7,k=4))}}
function xi(a,b,c){var d,e=-1,f=b[1],g=0,h=7,k=4;0===f&&(h=138,k=3);for(d=0;d<=c;d++){var l=f;f=b[2*(d+1)+1];if(!(++g<h&&l===f)){if(g<k){do Ph(a,l,a.S);while(0!==--g)}else 0!==l?(l!==e&&(Ph(a,l,a.S),g--),Ph(a,16,a.S),Oh(a,g-3,2)):10>=g?(Ph(a,17,a.S),Oh(a,g-3,3)):(Ph(a,18,a.S),Oh(a,g-11,7));g=0;e=l;0===f?(h=138,k=3):l===f?(h=6,k=3):(h=7,k=4)}}}
function yi(a){var b=4093624447,c;for(c=0;31>=c;c++,b>>>=1)if(b&1&&0!==a.X[2*c])return 0;if(0!==a.X[18]||0!==a.X[20]||0!==a.X[26])return 1;for(c=32;256>c;c++)if(0!==a.X[2*c])return 1;return 0}
var zi=!1;function Ai(a,b,c){a.L[a.jb+2*a.da]=b>>>8&255;a.L[a.jb+2*a.da+1]=b&255;a.L[a.Pb+a.da]=c&255;a.da++;0===b?a.X[2*c]++:(a.matches++,b--,a.X[2*(Fh[c]+256+1)]++,a.Ia[2*(256>b?Eh[b]:Eh[256+(b>>>7)])]++);return a.da===a.nb-1}
;function Bi(a,b){a.msg=wh[b];return b}
function Ci(a){for(var b=a.length;0<=--b;)a[b]=0}
function Di(a){var b=a.state,c=b.pending;c>a.F&&(c=a.F);0!==c&&(N.Qa(a.output,b.L,b.ob,c,a.Ya),a.Ya+=c,b.ob+=c,a.cc+=c,a.F-=c,b.pending-=c,0===b.pending&&(b.ob=0))}
function Ei(a,b){var c=0<=a.Z?a.Z:-1,d=a.j-a.Z,e=0;if(0<a.level){2===a.C.Kb&&(a.C.Kb=yi(a));vi(a,a.Cb);vi(a,a.yb);wi(a,a.X,a.Cb.Xa);wi(a,a.Ia,a.yb.Xa);vi(a,a.hc);for(e=18;3<=e&&0===a.S[2*Bh[e]+1];e--);a.xa+=3*(e+1)+14;var f=a.xa+3+7>>>3;var g=a.bb+3+7>>>3;g<=f&&(f=g)}else f=g=d+5;if(d+4<=f&&-1!==c)Oh(a,b?1:0,3),ri(a,c,d);else if(4===a.strategy||g===f)Oh(a,2+(b?1:0),3),ui(a,Ch,Dh);else{Oh(a,4+(b?1:0),3);c=a.Cb.Xa+1;d=a.yb.Xa+1;e+=1;Oh(a,c-257,5);Oh(a,d-1,5);Oh(a,e-4,4);for(f=0;f<e;f++)Oh(a,a.S[2*Bh[f]+
1],3);xi(a,a.X,c-1);xi(a,a.Ia,d-1);ui(a,a.X,a.Ia)}pi(a);b&&qi(a);a.Z=a.j;Di(a.C)}
function Q(a,b){a.L[a.pending++]=b}
function Fi(a,b){a.L[a.pending++]=b>>>8&255;a.L[a.pending++]=b&255}
function Gi(a,b){var c=a.yc,d=a.j,e=a.aa,f=a.zc,g=a.j>a.T-262?a.j-(a.T-262):0,h=a.window,k=a.Fa,l=a.la,n=a.j+258,m=h[d+e-1],q=h[d+e];a.aa>=a.pc&&(c>>=2);f>a.m&&(f=a.m);do{var p=b;if(h[p+e]===q&&h[p+e-1]===m&&h[p]===h[d]&&h[++p]===h[d+1]){d+=2;for(p++;h[++d]===h[++p]&&h[++d]===h[++p]&&h[++d]===h[++p]&&h[++d]===h[++p]&&h[++d]===h[++p]&&h[++d]===h[++p]&&h[++d]===h[++p]&&h[++d]===h[++p]&&d<n;);p=258-(n-d);d=n-258;if(p>e){a.Wa=b;e=p;if(p>=f)break;m=h[d+e-1];q=h[d+e]}}}while((b=l[b&k])>g&&0!==--c);return e<=
a.m?e:a.m}
function Hi(a){var b=a.T,c;do{var d=a.Jc-a.m-a.j;if(a.j>=b+(b-262)){N.Qa(a.window,a.window,b,b,0);a.Wa-=b;a.j-=b;a.Z-=b;var e=c=a.Bb;do{var f=a.head[--e];a.head[e]=f>=b?f-b:0}while(--c);e=c=b;do f=a.la[--e],a.la[e]=f>=b?f-b:0;while(--c);d+=b}if(0===a.C.U)break;e=a.C;c=a.window;f=a.j+a.m;var g=e.U;g>d&&(g=d);0===g?c=0:(e.U-=g,N.Qa(c,e.input,e.Na,g,f),1===e.state.wrap?e.A=qh(e.A,c,g,f):2===e.state.wrap&&(e.A=rh(e.A,c,g,f)),e.Na+=g,e.Oa+=g,c=g);a.m+=c;if(3<=a.m+a.Y)for(d=a.j-a.Y,a.D=a.window[d],a.D=
(a.D<<a.va^a.window[d+1])&a.ta;a.Y&&!(a.D=(a.D<<a.va^a.window[d+3-1])&a.ta,a.la[d&a.Fa]=a.head[a.D],a.head[a.D]=d,d++,a.Y--,3>a.m+a.Y););}while(262>a.m&&0!==a.C.U)}
function Ii(a,b){for(var c;;){if(262>a.m){Hi(a);if(262>a.m&&0===b)return 1;if(0===a.m)break}c=0;3<=a.m&&(a.D=(a.D<<a.va^a.window[a.j+3-1])&a.ta,c=a.la[a.j&a.Fa]=a.head[a.D],a.head[a.D]=a.j);0!==c&&a.j-c<=a.T-262&&(a.H=Gi(a,c));if(3<=a.H)if(c=Ai(a,a.j-a.Wa,a.H-3),a.m-=a.H,a.H<=a.Rb&&3<=a.m){a.H--;do a.j++,a.D=(a.D<<a.va^a.window[a.j+3-1])&a.ta,a.la[a.j&a.Fa]=a.head[a.D],a.head[a.D]=a.j;while(0!==--a.H);a.j++}else a.j+=a.H,a.H=0,a.D=a.window[a.j],a.D=(a.D<<a.va^a.window[a.j+1])&a.ta;else c=Ai(a,0,a.window[a.j]),
a.m--,a.j++;if(c&&(Ei(a,!1),0===a.C.F))return 1}a.Y=2>a.j?a.j:2;return 4===b?(Ei(a,!0),0===a.C.F?3:4):a.da&&(Ei(a,!1),0===a.C.F)?1:2}
function Ji(a,b){for(var c,d;;){if(262>a.m){Hi(a);if(262>a.m&&0===b)return 1;if(0===a.m)break}c=0;3<=a.m&&(a.D=(a.D<<a.va^a.window[a.j+3-1])&a.ta,c=a.la[a.j&a.Fa]=a.head[a.D],a.head[a.D]=a.j);a.aa=a.H;a.Cc=a.Wa;a.H=2;0!==c&&a.aa<a.Rb&&a.j-c<=a.T-262&&(a.H=Gi(a,c),5>=a.H&&(1===a.strategy||3===a.H&&4096<a.j-a.Wa)&&(a.H=2));if(3<=a.aa&&a.H<=a.aa){d=a.j+a.m-3;c=Ai(a,a.j-1-a.Cc,a.aa-3);a.m-=a.aa-1;a.aa-=2;do++a.j<=d&&(a.D=(a.D<<a.va^a.window[a.j+3-1])&a.ta,a.la[a.j&a.Fa]=a.head[a.D],a.head[a.D]=a.j);while(0!==
--a.aa);a.La=0;a.H=2;a.j++;if(c&&(Ei(a,!1),0===a.C.F))return 1}else if(a.La){if((c=Ai(a,0,a.window[a.j-1]))&&Ei(a,!1),a.j++,a.m--,0===a.C.F)return 1}else a.La=1,a.j++,a.m--}a.La&&(Ai(a,0,a.window[a.j-1]),a.La=0);a.Y=2>a.j?a.j:2;return 4===b?(Ei(a,!0),0===a.C.F?3:4):a.da&&(Ei(a,!1),0===a.C.F)?1:2}
function Ki(a,b){for(var c,d,e,f=a.window;;){if(258>=a.m){Hi(a);if(258>=a.m&&0===b)return 1;if(0===a.m)break}a.H=0;if(3<=a.m&&0<a.j&&(d=a.j-1,c=f[d],c===f[++d]&&c===f[++d]&&c===f[++d])){for(e=a.j+258;c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&d<e;);a.H=258-(e-d);a.H>a.m&&(a.H=a.m)}3<=a.H?(c=Ai(a,1,a.H-3),a.m-=a.H,a.j+=a.H,a.H=0):(c=Ai(a,0,a.window[a.j]),a.m--,a.j++);if(c&&(Ei(a,!1),0===a.C.F))return 1}a.Y=0;return 4===b?(Ei(a,!0),0===a.C.F?3:4):
a.da&&(Ei(a,!1),0===a.C.F)?1:2}
function Li(a,b){for(var c;;){if(0===a.m&&(Hi(a),0===a.m)){if(0===b)return 1;break}a.H=0;c=Ai(a,0,a.window[a.j]);a.m--;a.j++;if(c&&(Ei(a,!1),0===a.C.F))return 1}a.Y=0;return 4===b?(Ei(a,!0),0===a.C.F?3:4):a.da&&(Ei(a,!1),0===a.C.F)?1:2}
function Mi(a,b,c,d,e){this.dd=a;this.qd=b;this.td=c;this.pd=d;this.bd=e}
var Ni;Ni=[new Mi(0,0,0,0,function(a,b){var c=65535;for(c>a.ea-5&&(c=a.ea-5);;){if(1>=a.m){Hi(a);if(0===a.m&&0===b)return 1;if(0===a.m)break}a.j+=a.m;a.m=0;var d=a.Z+c;if(0===a.j||a.j>=d)if(a.m=a.j-d,a.j=d,Ei(a,!1),0===a.C.F)return 1;if(a.j-a.Z>=a.T-262&&(Ei(a,!1),0===a.C.F))return 1}a.Y=0;if(4===b)return Ei(a,!0),0===a.C.F?3:4;a.j>a.Z&&Ei(a,!1);return 1}),
new Mi(4,4,8,4,Ii),new Mi(4,5,16,8,Ii),new Mi(4,6,32,32,Ii),new Mi(4,4,16,16,Ji),new Mi(8,16,32,32,Ji),new Mi(8,16,128,128,Ji),new Mi(8,32,128,256,Ji),new Mi(32,128,258,1024,Ji),new Mi(32,258,258,4096,Ji)];
function Oi(){this.C=null;this.status=0;this.L=null;this.wrap=this.pending=this.ob=this.ea=0;this.v=null;this.fa=0;this.method=8;this.Ua=-1;this.Fa=this.dc=this.T=0;this.window=null;this.Jc=0;this.head=this.la=null;this.zc=this.pc=this.strategy=this.level=this.Rb=this.yc=this.aa=this.m=this.Wa=this.j=this.La=this.Cc=this.H=this.Z=this.va=this.ta=this.Lb=this.Bb=this.D=0;this.X=new N.pa(1146);this.Ia=new N.pa(122);this.S=new N.pa(78);Ci(this.X);Ci(this.Ia);Ci(this.S);this.hc=this.yb=this.Cb=null;this.ra=
new N.pa(16);this.M=new N.pa(573);Ci(this.M);this.Ta=this.wa=0;this.depth=new N.pa(573);Ci(this.depth);this.P=this.V=this.Y=this.matches=this.bb=this.xa=this.jb=this.da=this.nb=this.Pb=0}
function Pi(a,b){if(!a||!a.state||5<b||0>b)return a?Bi(a,-2):-2;var c=a.state;if(!a.output||!a.input&&0!==a.U||666===c.status&&4!==b)return Bi(a,0===a.F?-5:-2);c.C=a;var d=c.Ua;c.Ua=b;if(42===c.status)if(2===c.wrap)a.A=0,Q(c,31),Q(c,139),Q(c,8),c.v?(Q(c,(c.v.text?1:0)+(c.v.Ba?2:0)+(c.v.extra?4:0)+(c.v.name?8:0)+(c.v.comment?16:0)),Q(c,c.v.time&255),Q(c,c.v.time>>8&255),Q(c,c.v.time>>16&255),Q(c,c.v.time>>24&255),Q(c,9===c.level?2:2<=c.strategy||2>c.level?4:0),Q(c,c.v.xe&255),c.v.extra&&c.v.extra.length&&
(Q(c,c.v.extra.length&255),Q(c,c.v.extra.length>>8&255)),c.v.Ba&&(a.A=rh(a.A,c.L,c.pending,0)),c.fa=0,c.status=69):(Q(c,0),Q(c,0),Q(c,0),Q(c,0),Q(c,0),Q(c,9===c.level?2:2<=c.strategy||2>c.level?4:0),Q(c,3),c.status=113);else{var e=8+(c.dc-8<<4)<<8;e|=(2<=c.strategy||2>c.level?0:6>c.level?1:6===c.level?2:3)<<6;0!==c.j&&(e|=32);c.status=113;Fi(c,e+(31-e%31));0!==c.j&&(Fi(c,a.A>>>16),Fi(c,a.A&65535));a.A=1}if(69===c.status)if(c.v.extra){for(e=c.pending;c.fa<(c.v.extra.length&65535)&&(c.pending!==c.ea||
(c.v.Ba&&c.pending>e&&(a.A=rh(a.A,c.L,c.pending-e,e)),Di(a),e=c.pending,c.pending!==c.ea));)Q(c,c.v.extra[c.fa]&255),c.fa++;c.v.Ba&&c.pending>e&&(a.A=rh(a.A,c.L,c.pending-e,e));c.fa===c.v.extra.length&&(c.fa=0,c.status=73)}else c.status=73;if(73===c.status)if(c.v.name){e=c.pending;do{if(c.pending===c.ea&&(c.v.Ba&&c.pending>e&&(a.A=rh(a.A,c.L,c.pending-e,e)),Di(a),e=c.pending,c.pending===c.ea)){var f=1;break}f=c.fa<c.v.name.length?c.v.name.charCodeAt(c.fa++)&255:0;Q(c,f)}while(0!==f);c.v.Ba&&c.pending>
e&&(a.A=rh(a.A,c.L,c.pending-e,e));0===f&&(c.fa=0,c.status=91)}else c.status=91;if(91===c.status)if(c.v.comment){e=c.pending;do{if(c.pending===c.ea&&(c.v.Ba&&c.pending>e&&(a.A=rh(a.A,c.L,c.pending-e,e)),Di(a),e=c.pending,c.pending===c.ea)){f=1;break}f=c.fa<c.v.comment.length?c.v.comment.charCodeAt(c.fa++)&255:0;Q(c,f)}while(0!==f);c.v.Ba&&c.pending>e&&(a.A=rh(a.A,c.L,c.pending-e,e));0===f&&(c.status=103)}else c.status=103;103===c.status&&(c.v.Ba?(c.pending+2>c.ea&&Di(a),c.pending+2<=c.ea&&(Q(c,a.A&
255),Q(c,a.A>>8&255),a.A=0,c.status=113)):c.status=113);if(0!==c.pending){if(Di(a),0===a.F)return c.Ua=-1,0}else if(0===a.U&&(b<<1)-(4<b?9:0)<=(d<<1)-(4<d?9:0)&&4!==b)return Bi(a,-5);if(666===c.status&&0!==a.U)return Bi(a,-5);if(0!==a.U||0!==c.m||0!==b&&666!==c.status){d=2===c.strategy?Li(c,b):3===c.strategy?Ki(c,b):Ni[c.level].bd(c,b);if(3===d||4===d)c.status=666;if(1===d||3===d)return 0===a.F&&(c.Ua=-1),0;if(2===d&&(1===b?(Oh(c,2,3),Ph(c,256,Ch),16===c.P?(Nh(c,c.V),c.V=0,c.P=0):8<=c.P&&(c.L[c.pending++]=
c.V&255,c.V>>=8,c.P-=8)):5!==b&&(Oh(c,0,3),ri(c,0,0),3===b&&(Ci(c.head),0===c.m&&(c.j=0,c.Z=0,c.Y=0))),Di(a),0===a.F))return c.Ua=-1,0}if(4!==b)return 0;if(0>=c.wrap)return 1;2===c.wrap?(Q(c,a.A&255),Q(c,a.A>>8&255),Q(c,a.A>>16&255),Q(c,a.A>>24&255),Q(c,a.Oa&255),Q(c,a.Oa>>8&255),Q(c,a.Oa>>16&255),Q(c,a.Oa>>24&255)):(Fi(c,a.A>>>16),Fi(c,a.A&65535));Di(a);0<c.wrap&&(c.wrap=-c.wrap);return 0!==c.pending?0:1}
;var Qi={};Qi=function(){this.input=null;this.Oa=this.U=this.Na=0;this.output=null;this.cc=this.F=this.Ya=0;this.msg="";this.state=null;this.Kb=2;this.A=0};var Ri=Object.prototype.toString;
function Si(a){if(!(this instanceof Si))return new Si(a);a=this.options=N.assign({level:-1,method:8,chunkSize:16384,Ga:15,sd:8,strategy:0,Ea:""},a||{});a.raw&&0<a.Ga?a.Ga=-a.Ga:a.ed&&0<a.Ga&&16>a.Ga&&(a.Ga+=16);this.err=0;this.msg="";this.ended=!1;this.chunks=[];this.C=new Qi;this.C.F=0;var b=this.C;var c=a.level,d=a.method,e=a.Ga,f=a.sd,g=a.strategy;if(b){var h=1;-1===c&&(c=6);0>e?(h=0,e=-e):15<e&&(h=2,e-=16);if(1>f||9<f||8!==d||8>e||15<e||0>c||9<c||0>g||4<g)b=Bi(b,-2);else{8===e&&(e=9);var k=new Oi;
b.state=k;k.C=b;k.wrap=h;k.v=null;k.dc=e;k.T=1<<k.dc;k.Fa=k.T-1;k.Lb=f+7;k.Bb=1<<k.Lb;k.ta=k.Bb-1;k.va=~~((k.Lb+3-1)/3);k.window=new N.Pa(2*k.T);k.head=new N.pa(k.Bb);k.la=new N.pa(k.T);k.nb=1<<f+6;k.ea=4*k.nb;k.L=new N.Pa(k.ea);k.jb=1*k.nb;k.Pb=3*k.nb;k.level=c;k.strategy=g;k.method=d;if(b&&b.state){b.Oa=b.cc=0;b.Kb=2;c=b.state;c.pending=0;c.ob=0;0>c.wrap&&(c.wrap=-c.wrap);c.status=c.wrap?42:113;b.A=2===c.wrap?0:1;c.Ua=0;if(!zi){d=Array(16);for(f=g=0;28>f;f++)for(Gh[f]=g,e=0;e<1<<yh[f];e++)Fh[g++]=
f;Fh[g-1]=f;for(f=g=0;16>f;f++)for(Hh[f]=g,e=0;e<1<<zh[f];e++)Eh[g++]=f;for(g>>=7;30>f;f++)for(Hh[f]=g<<7,e=0;e<1<<zh[f]-7;e++)Eh[256+g++]=f;for(e=0;15>=e;e++)d[e]=0;for(e=0;143>=e;)Ch[2*e+1]=8,e++,d[8]++;for(;255>=e;)Ch[2*e+1]=9,e++,d[9]++;for(;279>=e;)Ch[2*e+1]=7,e++,d[7]++;for(;287>=e;)Ch[2*e+1]=8,e++,d[8]++;oi(Ch,287,d);for(e=0;30>e;e++)Dh[2*e+1]=5,Dh[2*e]=Qh(e,5);Jh=new Ih(Ch,yh,257,286,15);Kh=new Ih(Dh,zh,0,30,15);Lh=new Ih([],Ah,0,19,7);zi=!0}c.Cb=new Mh(c.X,Jh);c.yb=new Mh(c.Ia,Kh);c.hc=new Mh(c.S,
Lh);c.V=0;c.P=0;pi(c);c=0}else c=Bi(b,-2);0===c&&(b=b.state,b.Jc=2*b.T,Ci(b.head),b.Rb=Ni[b.level].qd,b.pc=Ni[b.level].dd,b.zc=Ni[b.level].td,b.yc=Ni[b.level].pd,b.j=0,b.Z=0,b.m=0,b.Y=0,b.H=b.aa=2,b.La=0,b.D=0);b=c}}else b=-2;if(0!==b)throw Error(wh[b]);a.header&&(b=this.C)&&b.state&&2===b.state.wrap&&(b.state.v=a.header);if(a.kb){var l;"string"===typeof a.kb?l=ph(a.kb):"[object ArrayBuffer]"===Ri.call(a.kb)?l=new Uint8Array(a.kb):l=a.kb;a=this.C;f=l;g=f.length;if(a&&a.state)if(l=a.state,b=l.wrap,
2===b||1===b&&42!==l.status||l.m)b=-2;else{1===b&&(a.A=qh(a.A,f,g,0));l.wrap=0;g>=l.T&&(0===b&&(Ci(l.head),l.j=0,l.Z=0,l.Y=0),c=new N.Pa(l.T),N.Qa(c,f,g-l.T,l.T,0),f=c,g=l.T);c=a.U;d=a.Na;e=a.input;a.U=g;a.Na=0;a.input=f;for(Hi(l);3<=l.m;){f=l.j;g=l.m-2;do l.D=(l.D<<l.va^l.window[f+3-1])&l.ta,l.la[f&l.Fa]=l.head[l.D],l.head[l.D]=f,f++;while(--g);l.j=f;l.m=2;Hi(l)}l.j+=l.m;l.Z=l.j;l.Y=l.m;l.m=0;l.H=l.aa=2;l.La=0;a.Na=d;a.input=e;a.U=c;l.wrap=b;b=0}else b=-2;if(0!==b)throw Error(wh[b]);this.me=!0}}
Si.prototype.push=function(a,b){var c=this.C,d=this.options.chunkSize;if(this.ended)return!1;var e=b===~~b?b:!0===b?4:0;"string"===typeof a?c.input=ph(a):"[object ArrayBuffer]"===Ri.call(a)?c.input=new Uint8Array(a):c.input=a;c.Na=0;c.U=c.input.length;do{0===c.F&&(c.output=new N.Pa(d),c.Ya=0,c.F=d);a=Pi(c,e);if(1!==a&&0!==a)return Ti(this,a),this.ended=!0,!1;if(0===c.F||0===c.U&&(4===e||2===e))if("string"===this.options.Ea){var f=N.ac(c.output,c.Ya);b=f;f=f.length;if(65537>f&&(b.subarray&&oh||!b.subarray))b=
String.fromCharCode.apply(null,N.ac(b,f));else{for(var g="",h=0;h<f;h++)g+=String.fromCharCode(b[h]);b=g}this.chunks.push(b)}else b=N.ac(c.output,c.Ya),this.chunks.push(b)}while((0<c.U||0===c.F)&&1!==a);if(4===e)return(c=this.C)&&c.state?(d=c.state.status,42!==d&&69!==d&&73!==d&&91!==d&&103!==d&&113!==d&&666!==d?a=Bi(c,-2):(c.state=null,a=113===d?Bi(c,-3):0)):a=-2,Ti(this,a),this.ended=!0,0===a;2===e&&(Ti(this,0),c.F=0);return!0};
function Ti(a,b){0===b&&(a.result="string"===a.options.Ea?a.chunks.join(""):N.nc(a.chunks));a.chunks=[];a.err=b;a.msg=a.C.msg}
;function Ui(a){this.name=a}
;var Vi=new Ui("rawColdConfigGroup");var Wi=new Ui("rawHotConfigGroup");function Xi(a){this.s=I(a)}
y(Xi,L);Xi.prototype.g=function(a){K(this,5,a)};function Yi(a){this.s=I(a)}
y(Yi,L);function Zi(a){this.s=I(a)}
y(Zi,L);Zi.ma=[2];function $i(a){this.s=I(a)}
y($i,L);$i.prototype.getPlayerType=function(){var a=0;a=void 0===a?0:a;var b=Qd(this,36);b=null==b?b:Number.isFinite(b)?b|0:void 0;return null!=b?b:a};
$i.prototype.setHomeGroupInfo=function(a){return J(this,Zi,81,a)};
$i.ma=[9,66,32,86,100,101];function aj(a){this.s=I(a)}
y(aj,L);aj.prototype.getKey=function(){return be(this,1)};
aj.prototype.ja=function(){return be(this,Xd(this,bj,2))};
var bj=[2,3,4,5,6];function cj(a){this.s=I(a)}
y(cj,L);cj.ma=[15,26,28];function dj(a){this.s=I(a)}
y(dj,L);dj.ma=[5];function ej(a){this.s=I(a)}
y(ej,L);function fj(a){this.s=I(a)}
y(fj,L);fj.prototype.setSafetyMode=function(a){return fe(this,5,a)};
fj.ma=[12];function gj(a){this.s=I(a)}
y(gj,L);gj.ma=[12];var hj={le:"WEB_DISPLAY_MODE_UNKNOWN",he:"WEB_DISPLAY_MODE_BROWSER",je:"WEB_DISPLAY_MODE_MINIMAL_UI",ke:"WEB_DISPLAY_MODE_STANDALONE",ie:"WEB_DISPLAY_MODE_FULLSCREEN"};function ij(a){this.s=I(a)}
y(ij,L);ij.prototype.getKey=function(){return be(this,1)};
ij.prototype.ja=function(){return be(this,2)};function jj(a){this.s=I(a)}
y(jj,L);jj.ma=[4,5];function kj(a){this.s=I(a)}
y(kj,L);function lj(a){this.s=I(a)}
y(lj,L);var mj=[2,3,4,5];function nj(a){this.s=I(a)}
y(nj,L);function oj(a){this.s=I(a)}
y(oj,L);function pj(a){this.s=I(a)}
y(pj,L);function qj(a){this.s=I(a)}
y(qj,L);qj.ma=[10,17];function rj(a){this.s=I(a)}
y(rj,L);function sj(a){this.s=I(a)}
y(sj,L);function tj(a){this.s=I(a)}
y(tj,L);function uj(a){this.s=I(a,496)}
y(uj,L);
var vj=[2,3,5,6,7,11,13,20,21,22,23,24,28,32,37,45,59,72,73,74,76,78,79,80,85,91,97,100,102,105,111,117,119,126,127,136,146,148,151,156,157,158,159,163,164,168,176,177,178,179,184,188,189,190,191,193,194,195,196,197,198,199,200,201,202,203,204,205,206,208,209,215,219,222,225,226,227,229,232,233,234,240,241,244,247,248,249,251,254,255,256,257,258,259,260,261,266,270,272,278,288,291,293,300,304,308,309,310,311,313,314,319,320,321,323,324,327,328,330,331,332,334,337,338,340,344,348,350,351,352,353,354,
355,356,357,358,361,363,364,368,369,370,373,374,375,378,380,381,383,388,389,399,402,403,410,411,412,413,414,415,416,417,418,423,424,425,426,427,429,430,431,439,441,444,448,458,469,471,473,474,480,481,482,484,485,486,491,495];function wj(a){this.s=I(a)}
y(wj,L);function xj(a){this.s=I(a)}
y(xj,L);xj.prototype.getPlaylistId=function(){return ce(this,2)};
var de=[1,2];function yj(a){this.s=I(a)}
y(yj,L);yj.ma=[3];var zj=D.window,Aj,Bj,Cj=(null==zj?void 0:null==(Aj=zj.yt)?void 0:Aj.config_)||(null==zj?void 0:null==(Bj=zj.ytcfg)?void 0:Bj.data_)||{};F("yt.config_",Cj);function Dj(){var a=arguments;1<a.length?Cj[a[0]]=a[1]:1===a.length&&Object.assign(Cj,a[0])}
function R(a,b){return a in Cj?Cj[a]:b}
function Ej(){return R("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS")}
function Fj(){var a=Cj.EXPERIMENT_FLAGS;return a?a.web_disable_gel_stp_ecatcher_killswitch:void 0}
;var Gj=[];function Hj(a){Gj.forEach(function(b){return b(a)})}
function Ij(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){Jj(b)}}:a}
function Jj(a){var b=E("yt.logging.errors.log");b?b(a,"ERROR",void 0,void 0,void 0,void 0,void 0):(b=R("ERRORS",[]),b.push([a,"ERROR",void 0,void 0,void 0,void 0,void 0]),Dj("ERRORS",b));Hj(a)}
function Kj(a,b,c,d,e){var f=E("yt.logging.errors.log");f?f(a,"WARNING",b,c,d,void 0,e):(f=R("ERRORS",[]),f.push([a,"WARNING",b,c,d,void 0,e]),Dj("ERRORS",f))}
;function S(a){a=Lj(a);return"string"===typeof a&&"false"===a?!1:!!a}
function T(a,b){a=Lj(a);return void 0===a&&void 0!==b?b:Number(a||0)}
function Lj(a){return R("EXPERIMENT_FLAGS",{})[a]}
function Mj(){for(var a=[],b=R("EXPERIMENTS_FORCED_FLAGS",{}),c=w(Object.keys(b)),d=c.next();!d.done;d=c.next())d=d.value,a.push({key:d,value:String(b[d])});c=R("EXPERIMENT_FLAGS",{});var e=w(Object.keys(c));for(d=e.next();!d.done;d=e.next())d=d.value,d.startsWith("force_")&&void 0===b[d]&&a.push({key:d,value:String(c[d])});return a}
;var Nj=0;F("ytDomDomGetNextId",E("ytDomDomGetNextId")||function(){return++Nj});var Oj={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function Pj(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.ctrlKey=this.altKey=!1;this.clientY=this.clientX=0;this.changedTouches=this.touches=null;try{if(a=a||window.event){this.event=a;for(var b in a)b in Oj||(this[b]=a[b]);var c=a.target||a.srcElement;c&&3==c.nodeType&&(c=c.parentNode);this.target=c;var d=a.relatedTarget;if(d)try{d=d.nodeName?d:null}catch(e){d=null}else"mouseover"==
this.type?d=a.fromElement:"mouseout"==this.type&&(d=a.toElement);this.relatedTarget=d;this.clientX=void 0!=a.clientX?a.clientX:a.pageX;this.clientY=void 0!=a.clientY?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||("keypress"==this.type?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey}}catch(e){}}
Pj.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
Pj.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
Pj.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var nb=D.ytEventsEventsListeners||{};F("ytEventsEventsListeners",nb);var Qj=D.ytEventsEventsCounter||{count:0};F("ytEventsEventsCounter",Qj);
function Rj(a,b,c,d){d=void 0===d?{}:d;a.addEventListener&&("mouseenter"!=b||"onmouseenter"in document?"mouseleave"!=b||"onmouseenter"in document?"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return mb(function(e){var f="boolean"===typeof e[4]&&e[4]==!!d,g=Oa(e[4])&&Oa(d)&&ob(e[4],d);return!!e.length&&e[0]==a&&e[1]==b&&e[2]==c&&(f||g)})}
function Sj(a){a&&("string"==typeof a&&(a=[a]),db(a,function(b){if(b in nb){var c=nb[b],d=c[0],e=c[1],f=c[3];c=c[4];d.removeEventListener?Tj()||"boolean"===typeof c?d.removeEventListener(e,f,c):d.removeEventListener(e,f,!!c.capture):d.detachEvent&&d.detachEvent("on"+e,f);delete nb[b]}}))}
var Tj=bb(function(){var a=!1;try{var b=Object.defineProperty({},"capture",{get:function(){a=!0}});
window.addEventListener("test",null,b)}catch(c){}return a});
function Uj(a,b,c){var d=void 0===d?{}:d;if(a&&(a.addEventListener||a.attachEvent)){var e=Rj(a,b,c,d);if(!e){e=++Qj.count+"";var f=!("mouseenter"!=b&&"mouseleave"!=b||!a.addEventListener||"onmouseenter"in document);var g=f?function(h){h=new Pj(h);if(!le(h.relatedTarget,function(k){return k==a}))return h.currentTarget=a,h.type=b,c.call(a,h)}:function(h){h=new Pj(h);
h.currentTarget=a;return c.call(a,h)};
g=Ij(g);a.addEventListener?("mouseenter"==b&&f?b="mouseover":"mouseleave"==b&&f?b="mouseout":"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),Tj()||"boolean"===typeof d?a.addEventListener(b,g,d):a.addEventListener(b,g,!!d.capture)):a.attachEvent("on"+b,g);nb[e]=[a,b,c,g,d]}}}
;function Vj(a,b){"function"===typeof a&&(a=Ij(a));return window.setTimeout(a,b)}
function Wj(a){"function"===typeof a&&(a=Ij(a));return window.setInterval(a,250)}
;var Xj=/^[\w.]*$/,Yj={q:!0,search_query:!0};function Zj(a,b){b=a.split(b);for(var c={},d=0,e=b.length;d<e;d++){var f=b[d].split("=");if(1==f.length&&f[0]||2==f.length)try{var g=ak(f[0]||""),h=ak(f[1]||"");g in c?Array.isArray(c[g])?kb(c[g],h):c[g]=[c[g],h]:c[g]=h}catch(m){var k=m,l=f[0],n=String(Zj);k.args=[{key:l,value:f[1],query:a,method:bk==n?"unchanged":n}];Yj.hasOwnProperty(l)||Kj(k)}}return c}
var bk=String(Zj);function ck(a){var b=[];lb(a,function(c,d){var e=encodeURIComponent(String(d)),f;Array.isArray(c)?f=c:f=[c];db(f,function(g){""==g?b.push(e):b.push(e+"="+encodeURIComponent(String(g)))})});
return b.join("&")}
function dk(a){"?"==a.charAt(0)&&(a=a.substr(1));return Zj(a,"&")}
function ek(a,b,c){var d=a.split("#",2);a=d[0];d=1<d.length?"#"+d[1]:"";var e=a.split("?",2);a=e[0];e=dk(e[1]||"");for(var f in b)!c&&null!==e&&f in e||(e[f]=b[f]);b=a;a=ec(e);a?(c=b.indexOf("#"),0>c&&(c=b.length),f=b.indexOf("?"),0>f||f>c?(f=c,e=""):e=b.substring(f+1,c),b=[b.slice(0,f),e,b.slice(c)],c=b[1],b[1]=a?c?c+"&"+a:a:c,a=b[0]+(b[1]?"?"+b[1]:"")+b[2]):a=b;return a+d}
function fk(a){if(!b)var b=window.location.href;var c=a.match($b)[1]||null,d=bc(a);c&&d?(a=a.match($b),b=b.match($b),a=a[3]==b[3]&&a[1]==b[1]&&a[4]==b[4]):a=d?bc(b)==d&&(Number(b.match($b)[4]||null)||null)==(Number(a.match($b)[4]||null)||null):!0;return a}
function ak(a){return a&&a.match(Xj)?a:decodeURIComponent(a.replace(/\+/g," "))}
;function gk(a){var b=hk;a=void 0===a?E("yt.ads.biscotti.lastId_")||"":a;var c=Object,d=c.assign,e={};e.dt=Ge;e.flash="0";a:{try{var f=b.g.top.location.href}catch(ia){f=2;break a}f=f?f===b.h.location.href?0:1:2}e=(e.frm=f,e);try{e.u_tz=-(new Date).getTimezoneOffset();var g=void 0===g?ie:g;try{var h=g.history.length}catch(ia){h=0}e.u_his=h;var k;e.u_h=null==(k=ie.screen)?void 0:k.height;var l;e.u_w=null==(l=ie.screen)?void 0:l.width;var n;e.u_ah=null==(n=ie.screen)?void 0:n.availHeight;var m;e.u_aw=
null==(m=ie.screen)?void 0:m.availWidth;var q;e.u_cd=null==(q=ie.screen)?void 0:q.colorDepth}catch(ia){}h=b.g;try{var p=h.screenX;var t=h.screenY}catch(ia){}try{var u=h.outerWidth;var z=h.outerHeight}catch(ia){}try{var C=h.innerWidth;var P=h.innerHeight}catch(ia){}try{var U=h.screenLeft;var Y=h.screenTop}catch(ia){}try{C=h.innerWidth,P=h.innerHeight}catch(ia){}try{var La=h.screen.availWidth;var Ca=h.screen.availTop}catch(ia){}p=[U,Y,p,t,La,Ca,u,z,C,P];t=b.g.top;try{var Ma=(t||window).document,ja=
"CSS1Compat"==Ma.compatMode?Ma.documentElement:Ma.body;var ra=(new je(ja.clientWidth,ja.clientHeight)).round()}catch(ia){ra=new je(-12245933,-12245933)}Ma=ra;ra={};var sa=void 0===sa?D:sa;ja=new Ef;"SVGElement"in sa&&"createElementNS"in sa.document&&ja.set(0);t=Ee();t["allow-top-navigation-by-user-activation"]&&ja.set(1);t["allow-popups-to-escape-sandbox"]&&ja.set(2);sa.crypto&&sa.crypto.subtle&&ja.set(3);"TextDecoder"in sa&&"TextEncoder"in sa&&ja.set(4);sa=Ff(ja);ra.bc=sa;ra.bih=Ma.height;ra.biw=
Ma.width;ra.brdim=p.join();b=b.h;b=(ra.vis=b.prerendering?3:{visible:1,hidden:2,prerender:3,preview:4,unloaded:5}[b.visibilityState||b.webkitVisibilityState||b.mozVisibilityState||""]||0,ra.wgl=!!ie.WebGLRenderingContext,ra);c=d.call(c,e,b);c.ca_type="image";a&&(c.bid=a);return c}
var hk=new function(){var a=window.document;this.g=window;this.h=a};
F("yt.ads_.signals_.getAdSignalsString",function(a){return ck(gk(a))});Wa();var ik="XMLHttpRequest"in D?function(){return new XMLHttpRequest}:null;
function jk(){if(!ik)return null;var a=ik();return"open"in a?a:null}
;var kk="client_dev_domain client_dev_expflag client_dev_regex_map client_dev_root_url client_rollout_override expflag forcedCapability jsfeat jsmode mods".split(" ");x(kk);var lk={Authorization:"AUTHORIZATION","X-Goog-EOM-Visitor-Id":"EOM_VISITOR_DATA","X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-Youtube-Domain-Admin-State":"DOMAIN_ADMIN_STATE","X-Youtube-Chrome-Connected":"CHROME_CONNECTED_HEADER","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-YouTube-Delegation-Context":"INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT","X-YouTube-Device":"DEVICE","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL",
"X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-YouTube-Variants-Checksum":"VARIANTS_CHECKSUM","X-Goog-AuthUser":"SESSION_INDEX","X-Goog-PageId":"DELEGATED_SESSION_ID"},mk="app debugcss debugjs expflag force_ad_params force_ad_encrypted force_viral_ad_response_params forced_experiments innertube_snapshots innertube_goldens internalcountrycode internalipoverride absolute_experiments conditional_experiments sbb sr_bns_address".split(" ").concat(x(kk)),nk=!1;
function ok(a,b){b=void 0===b?{}:b;var c=fk(a),d=S("web_ajax_ignore_global_headers_if_set"),e;for(e in lk){var f=R(lk[e]),g="X-Goog-AuthUser"===e||"X-Goog-PageId"===e;"X-Goog-Visitor-Id"!==e||f||(f=R("VISITOR_DATA"));!f||!c&&bc(a)||d&&void 0!==b[e]||"TVHTML5_UNPLUGGED"===R("INNERTUBE_CLIENT_NAME")&&g||(b[e]=f)}c&&R("WEBVIEW_EOM",!1)&&(b["X-Yt-Webview-Eom"]="1");"X-Goog-EOM-Visitor-Id"in b&&"X-Goog-Visitor-Id"in b&&delete b["X-Goog-Visitor-Id"];if(c||!bc(a))b["X-YouTube-Utc-Offset"]=String(-(new Date).getTimezoneOffset());
if(c||!bc(a)){try{var h=(new Intl.DateTimeFormat).resolvedOptions().timeZone}catch(k){}h&&(b["X-YouTube-Time-Zone"]=h)}document.location.hostname.endsWith("youtubeeducation.com")||!c&&bc(a)||(b["X-YouTube-Ad-Signals"]=ck(gk()));return b}
function pk(a){var b=window.location.search,c=bc(a);S("debug_handle_relative_url_for_query_forward_killswitch")||!c&&fk(a)&&(c=document.location.hostname);var d=ac(a.match($b)[5]||null);d=(c=c&&(c.endsWith("youtube.com")||c.endsWith("youtube-nocookie.com")))&&d&&d.startsWith("/api/");if(!c||d)return a;var e=dk(b),f={};db(mk,function(g){e[g]&&(f[g]=e[g])});
return ek(a,f||{},!1)}
function qk(a,b){var c=b.format||"JSON";a=rk(a,b);var d=sk(a,b),e=!1,f=tk(a,function(k){if(!e){e=!0;h&&window.clearTimeout(h);a:switch(k&&"status"in k?k.status:-1){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:var l=!0;break a;default:l=!1}var n=null,m=400<=k.status&&500>k.status,q=500<=k.status&&600>k.status;if(l||m||q)n=uk(a,c,k,b.convertToSafeHtml);if(l)a:if(k&&204==k.status)l=!0;else{switch(c){case "XML":l=0==parseInt(n&&n.return_code,10);break a;case "RAW":l=!0;break a}l=
!!n}n=n||{};m=b.context||D;l?b.onSuccess&&b.onSuccess.call(m,k,n):b.onError&&b.onError.call(m,k,n);b.onFinish&&b.onFinish.call(m,k,n)}},b.method,d,b.headers,b.responseType,b.withCredentials);
d=b.timeout||0;if(b.onTimeout&&0<d){var g=b.onTimeout;var h=Vj(function(){e||(e=!0,f.abort(),window.clearTimeout(h),g.call(b.context||D,f))},d)}return f}
function rk(a,b){b.includeDomain&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var c=R("XSRF_FIELD_NAME");if(b=b.urlParams)b[c]&&delete b[c],a=ek(a,b||{},!0);return a}
function sk(a,b){var c=R("XSRF_FIELD_NAME"),d=R("XSRF_TOKEN"),e=b.postBody||"",f=b.postParams,g=R("XSRF_FIELD_NAME"),h;b.headers&&(h=b.headers["Content-Type"]);b.excludeXsrf||bc(a)&&!b.withCredentials&&bc(a)!=document.location.hostname||"POST"!=b.method||h&&"application/x-www-form-urlencoded"!=h||b.postParams&&b.postParams[g]||(f||(f={}),f[c]=d);(S("ajax_parse_query_data_only_when_filled")&&f&&0<Object.keys(f).length||f)&&"string"===typeof e&&(e=dk(e),rb(e,f),e=b.postBodyFormat&&"JSON"==b.postBodyFormat?
JSON.stringify(e):ec(e));if(!(a=e)&&(a=f)){a:{for(var k in f){f=!1;break a}f=!0}a=!f}!nk&&a&&"POST"!=b.method&&(nk=!0,Jj(Error("AJAX request with postData should use POST")));return e}
function uk(a,b,c,d){var e=null;switch(b){case "JSON":try{var f=c.responseText}catch(g){throw d=Error("Error reading responseText"),d.params=a,Kj(d),g;}a=c.getResponseHeader("Content-Type")||"";f&&0<=a.indexOf("json")&&(")]}'\n"===f.substring(0,5)&&(f=f.substring(5)),e=JSON.parse(f));break;case "XML":if(a=(a=c.responseXML)?vk(a):null)e={},db(a.getElementsByTagName("*"),function(g){e[g.tagName]=wk(g)})}d&&xk(e);
return e}
function xk(a){if(Oa(a))for(var b in a){var c;(c="html_content"==b)||(c=b.length-5,c=0<=c&&b.indexOf("_html",c)==c);if(c){c=b;var d=a[b],e=tb();d=e?e.createHTML(d):d;a[c]=new Yb(d)}else xk(a[b])}}
function vk(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&0<a.length?a[0]:null:null}
function wk(a){var b="";db(a.childNodes,function(c){b+=c.nodeValue});
return b}
function yk(a,b){b.method="POST";b.postParams||(b.postParams={});return qk(a,b)}
function tk(a,b,c,d,e,f,g,h){function k(){4==(l&&"readyState"in l?l.readyState:0)&&b&&Ij(b)(l)}
c=void 0===c?"GET":c;d=void 0===d?"":d;h=void 0===h?!1:h;var l=jk();if(!l)return null;"onloadend"in l?l.addEventListener("loadend",k,!1):l.onreadystatechange=k;S("debug_forward_web_query_parameters")&&(a=pk(a));l.open(c,a,!0);f&&(l.responseType=f);g&&(l.withCredentials=!0);c="POST"==c&&(void 0===window.FormData||!(d instanceof FormData));if(e=ok(a,e))for(var n in e)l.setRequestHeader(n,e[n]),"content-type"==n.toLowerCase()&&(c=!1);c&&l.setRequestHeader("Content-Type","application/x-www-form-urlencoded");
h&&"setAttributionReporting"in XMLHttpRequest.prototype&&l.setAttributionReporting({eventSourceEligible:!0,triggerEligible:!1});l.send(d);return l}
;var zk=[{Sb:function(a){return"Cannot read property '"+a.key+"'"},
Db:{Error:[{regexp:/(Permission denied) to access property "([^']+)"/,groups:["reason","key"]}],TypeError:[{regexp:/Cannot read property '([^']+)' of (null|undefined)/,groups:["key","value"]},{regexp:/\u65e0\u6cd5\u83b7\u53d6\u672a\u5b9a\u4e49\u6216 (null|undefined) \u5f15\u7528\u7684\u5c5e\u6027\u201c([^\u201d]+)\u201d/,groups:["value","key"]},{regexp:/\uc815\uc758\ub418\uc9c0 \uc54a\uc74c \ub610\ub294 (null|undefined) \ucc38\uc870\uc778 '([^']+)' \uc18d\uc131\uc744 \uac00\uc838\uc62c \uc218 \uc5c6\uc2b5\ub2c8\ub2e4./,
groups:["value","key"]},{regexp:/No se puede obtener la propiedad '([^']+)' de referencia nula o sin definir/,groups:["key"]},{regexp:/Unable to get property '([^']+)' of (undefined or null) reference/,groups:["key","value"]},{regexp:/(null) is not an object \(evaluating '(?:([^.]+)\.)?([^']+)'\)/,groups:["value","base","key"]}]}},{Sb:function(a){return"Cannot call '"+a.key+"'"},
Db:{TypeError:[{regexp:/(?:([^ ]+)?\.)?([^ ]+) is not a function/,groups:["base","key"]},{regexp:/([^ ]+) called on (null or undefined)/,groups:["key","value"]},{regexp:/Object (.*) has no method '([^ ]+)'/,groups:["base","key"]},{regexp:/Object doesn't support property or method '([^ ]+)'/,groups:["key"]},{regexp:/\u30aa\u30d6\u30b8\u30a7\u30af\u30c8\u306f '([^']+)' \u30d7\u30ed\u30d1\u30c6\u30a3\u307e\u305f\u306f\u30e1\u30bd\u30c3\u30c9\u3092\u30b5\u30dd\u30fc\u30c8\u3057\u3066\u3044\u307e\u305b\u3093/,
groups:["key"]},{regexp:/\uac1c\uccb4\uac00 '([^']+)' \uc18d\uc131\uc774\ub098 \uba54\uc11c\ub4dc\ub97c \uc9c0\uc6d0\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4./,groups:["key"]}]}},{Sb:function(a){return a.key+" is not defined"},
Db:{ReferenceError:[{regexp:/(.*) is not defined/,groups:["key"]},{regexp:/Can't find variable: (.*)/,groups:["key"]}]}}];var Bk={Ca:[],za:[{ib:Ak,weight:500}]};function Ak(a){if("JavaException"===a.name)return!0;a=a.stack;return a.includes("chrome://")||a.includes("chrome-extension://")||a.includes("moz-extension://")}
;function Ck(){this.za=[];this.Ca=[]}
var Dk;function Ek(){if(!Dk){var a=Dk=new Ck;a.Ca.length=0;a.za.length=0;Bk.Ca&&a.Ca.push.apply(a.Ca,Bk.Ca);Bk.za&&a.za.push.apply(a.za,Bk.za)}return Dk}
;var Fk=new M;function Gk(a){function b(){return a.charCodeAt(d++)}
var c=a.length,d=0;do{var e=Hk(b);if(Infinity===e)break;var f=e>>3;switch(e&7){case 0:e=Hk(b);if(2===f)return e;break;case 1:if(2===f)return;d+=8;break;case 2:e=Hk(b);if(2===f)return a.substr(d,e);d+=e;break;case 5:if(2===f)return;d+=4;break;default:return}}while(d<c)}
function Hk(a){var b=a(),c=b&127;if(128>b)return c;b=a();c|=(b&127)<<7;if(128>b)return c;b=a();c|=(b&127)<<14;if(128>b)return c;b=a();return 128>b?c|(b&127)<<21:Infinity}
;function Ik(a,b,c,d){if(a)if(Array.isArray(a)){var e=d;for(d=0;d<a.length&&!(a[d]&&(e+=Jk(d,a[d],b,c),500<e));d++);d=e}else if("object"===typeof a)for(e in a){if(a[e]){var f=e;var g=a[e],h=b,k=c;f="string"!==typeof g||"clickTrackingParams"!==f&&"trackingParams"!==f?0:(g=Gk(atob(g.replace(/-/g,"+").replace(/_/g,"/"))))?Jk(f+".ve",g,h,k):0;d+=f;d+=Jk(e,a[e],b,c);if(500<d)break}}else c[b]=Kk(a),d+=c[b].length;else c[b]=Kk(a),d+=c[b].length;return d}
function Jk(a,b,c,d){c+="."+a;a=Kk(b);d[c]=a;return c.length+a.length}
function Kk(a){try{return("string"===typeof a?a:String(JSON.stringify(a))).substr(0,500)}catch(b){return"unable to serialize "+typeof a+" ("+b.message+")"}}
;function Lk(){}
;function Mk(){if(!D.matchMedia)return"WEB_DISPLAY_MODE_UNKNOWN";try{return D.matchMedia("(display-mode: standalone)").matches?"WEB_DISPLAY_MODE_STANDALONE":D.matchMedia("(display-mode: minimal-ui)").matches?"WEB_DISPLAY_MODE_MINIMAL_UI":D.matchMedia("(display-mode: fullscreen)").matches?"WEB_DISPLAY_MODE_FULLSCREEN":D.matchMedia("(display-mode: browser)").matches?"WEB_DISPLAY_MODE_BROWSER":"WEB_DISPLAY_MODE_UNKNOWN"}catch(a){return"WEB_DISPLAY_MODE_UNKNOWN"}}
;function Nk(a){switch(a){case "DESKTOP":return 1;case "UNKNOWN_PLATFORM":return 0;case "TV":return 2;case "GAME_CONSOLE":return 3;case "MOBILE":return 4;case "TABLET":return 5}}
;F("ytglobal.prefsUserPrefsPrefs_",E("ytglobal.prefsUserPrefsPrefs_")||{});var Ok={bluetooth:"CONN_DISCO",cellular:"CONN_CELLULAR_UNKNOWN",ethernet:"CONN_WIFI",none:"CONN_NONE",wifi:"CONN_WIFI",wimax:"CONN_CELLULAR_4G",other:"CONN_UNKNOWN",unknown:"CONN_UNKNOWN","slow-2g":"CONN_CELLULAR_2G","2g":"CONN_CELLULAR_2G","3g":"CONN_CELLULAR_3G","4g":"CONN_CELLULAR_4G"},Pk={CONN_DEFAULT:0,CONN_UNKNOWN:1,CONN_NONE:2,CONN_WIFI:3,CONN_CELLULAR_2G:4,CONN_CELLULAR_3G:5,CONN_CELLULAR_4G:6,CONN_CELLULAR_UNKNOWN:7,CONN_DISCO:8,CONN_CELLULAR_5G:9,CONN_WIFI_METERED:10,CONN_CELLULAR_5G_SA:11,
CONN_CELLULAR_5G_NSA:12,CONN_WIRED:30,CONN_INVALID:31},Qk={EFFECTIVE_CONNECTION_TYPE_UNKNOWN:0,EFFECTIVE_CONNECTION_TYPE_OFFLINE:1,EFFECTIVE_CONNECTION_TYPE_SLOW_2G:2,EFFECTIVE_CONNECTION_TYPE_2G:3,EFFECTIVE_CONNECTION_TYPE_3G:4,EFFECTIVE_CONNECTION_TYPE_4G:5},Rk={"slow-2g":"EFFECTIVE_CONNECTION_TYPE_SLOW_2G","2g":"EFFECTIVE_CONNECTION_TYPE_2G","3g":"EFFECTIVE_CONNECTION_TYPE_3G","4g":"EFFECTIVE_CONNECTION_TYPE_4G"};function Sk(){var a=D.navigator;return a?a.connection:void 0}
;function Tk(a){var b=Ga.apply(1,arguments);var c=Error.call(this,a);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.args=[].concat(x(b))}
y(Tk,Error);function Uk(){try{return Vk(),!0}catch(a){return!1}}
function Vk(){if(void 0!==R("DATASYNC_ID"))return R("DATASYNC_ID");throw new Tk("Datasync ID not set","unknown");}
;function Wk(){}
function Xk(a,b){return Df.Ha(a,0,b)}
Wk.prototype.qa=function(a,b){return this.Ha(a,1,b)};
Wk.prototype.gb=function(a){var b=E("yt.scheduler.instance.addImmediateJob");b?b(a):a()};var Yk=T("web_emulated_idle_callback_delay",300),Zk=1E3/60-3,$k=[8,5,4,3,2,1,0];
function al(a){a=void 0===a?{}:a;We.call(this);this.h=[];this.i={};this.eb=this.g=0;this.cb=this.u=!1;this.O=[];this.oa=this.fb=!1;for(var b=w($k),c=b.next();!c.done;c=b.next())this.h[c.value]=[];this.o=0;this.Qc=a.timeout||1;this.K=Zk;this.G=0;this.sb=this.ud.bind(this);this.Pc=this.zd.bind(this);this.Mc=this.Rc.bind(this);this.Nc=this.fd.bind(this);this.Oc=this.vd.bind(this);this.ec=!!window.requestIdleCallback&&!!window.cancelIdleCallback&&!S("disable_scheduler_requestIdleCallback");(this.ya=!1!==
a.useRaf&&!!window.requestAnimationFrame)&&document.addEventListener("visibilitychange",this.sb)}
y(al,We);r=al.prototype;r.gb=function(a){var b=Wa();bl(a);a=Wa()-b;this.u||(this.K-=a)};
r.Ha=function(a,b,c){++this.eb;if(10===b)return this.gb(a),this.eb;var d=this.eb;this.i[d]=a;this.u&&!c?this.O.push({id:d,priority:b}):(this.h[b].push(d),this.cb||this.u||(0!==this.g&&cl(this)!==this.G&&dl(this),this.start()));return d};
r.ba=function(a){delete this.i[a]};
function el(a){a.O.length=0;for(var b=5;0<=b;b--)a.h[b].length=0;a.h[8].length=0;a.i={};dl(a)}
function cl(a){if(a.h[8].length){if(a.oa)return 4;if(!document.hidden&&a.ya)return 3}for(var b=5;b>=a.o;b--)if(0<a.h[b].length)return 0<b?!document.hidden&&a.ya?3:2:1;return 0}
function fl(a){var b=E("yt.logging.errors.log");b&&b(a)}
function bl(a){try{a()}catch(b){fl(b)}}
function gl(a){for(var b=w($k),c=b.next();!c.done;c=b.next())if(a.h[c.value].length)return!0;return!1}
r.fd=function(a){var b=void 0;a&&(b=a.timeRemaining());this.fb=!0;hl(this,b);this.fb=!1};
r.zd=function(){hl(this)};
r.Rc=function(){il(this)};
r.vd=function(a){this.oa=!0;var b=cl(this);4===b&&b!==this.G&&(dl(this),this.start());hl(this,void 0,a);this.oa=!1};
r.ud=function(){document.hidden||il(this);this.g&&(dl(this),this.start())};
function il(a){dl(a);a.u=!0;for(var b=Wa(),c=a.h[8];c.length;){var d=c.shift(),e=a.i[d];delete a.i[d];e&&bl(e)}jl(a);a.u=!1;gl(a)&&a.start();b=Wa()-b;a.K-=b}
function jl(a){for(var b=0,c=a.O.length;b<c;b++){var d=a.O[b];a.h[d.priority].push(d.id)}a.O.length=0}
function hl(a,b,c){a.oa&&4===a.G&&a.g||dl(a);a.u=!0;b=Wa()+(b||a.K);for(var d=a.h[5];d.length;){var e=d.shift(),f=a.i[e];delete a.i[e];if(f)try{f(c)}catch(l){fl(l)}}for(d=a.h[4];d.length;)c=d.shift(),e=a.i[c],delete a.i[c],e&&bl(e);d=a.fb?0:1;d=a.o>d?a.o:d;if(!(Wa()>=b)){do{a:{c=a;e=d;for(f=3;f>=e;f--)for(var g=c.h[f];g.length;){var h=g.shift(),k=c.i[h];delete c.i[h];if(k){c=k;break a}}c=null}c&&bl(c)}while(c&&Wa()<b)}a.u=!1;jl(a);a.K=Zk;gl(a)&&a.start()}
r.start=function(){this.cb=!1;if(0===this.g)switch(this.G=cl(this),this.G){case 1:var a=this.Nc;this.g=this.ec?window.requestIdleCallback(a,{timeout:3E3}):window.setTimeout(a,Yk);break;case 2:this.g=window.setTimeout(this.Pc,this.Qc);break;case 3:this.g=window.requestAnimationFrame(this.Oc);break;case 4:this.g=window.setTimeout(this.Mc,0)}};
function dl(a){if(a.g){switch(a.G){case 1:var b=a.g;a.ec?window.cancelIdleCallback(b):window.clearTimeout(b);break;case 2:case 4:window.clearTimeout(a.g);break;case 3:window.cancelAnimationFrame(a.g)}a.g=0}}
r.sa=function(){el(this);dl(this);this.ya&&document.removeEventListener("visibilitychange",this.sb);We.prototype.sa.call(this)};var kl=E("yt.scheduler.instance.timerIdMap_")||{},ll=T("kevlar_tuner_scheduler_soft_state_timer_ms",800),ml=0,nl=0;function ol(){var a=E("ytglobal.schedulerInstanceInstance_");if(!a||a.Aa)a=new al(R("scheduler")||{}),F("ytglobal.schedulerInstanceInstance_",a);return a}
function pl(){ql();var a=E("ytglobal.schedulerInstanceInstance_");a&&(Se(a),F("ytglobal.schedulerInstanceInstance_",null))}
function ql(){el(ol());for(var a in kl)kl.hasOwnProperty(a)&&delete kl[Number(a)]}
function rl(a,b,c){if(!c)return c=void 0===c,-ol().Ha(a,b,c);var d=window.setTimeout(function(){var e=ol().Ha(a,b);kl[d]=e},c);
return d}
function sl(a){ol().gb(a)}
function tl(a){var b=ol();if(0>a)b.ba(-a);else{var c=kl[a];c?(b.ba(c),delete kl[a]):window.clearTimeout(a)}}
function ul(){vl()}
function vl(){window.clearTimeout(ml);ol().start()}
function wl(){var a=ol();dl(a);a.cb=!0;window.clearTimeout(ml);ml=window.setTimeout(ul,ll)}
function xl(){window.clearTimeout(nl);nl=window.setTimeout(function(){yl(0)},ll)}
function yl(a){xl();var b=ol();b.o=a;b.start()}
function zl(a){xl();var b=ol();b.o>a&&(b.o=a,b.start())}
function Al(){window.clearTimeout(nl);var a=ol();a.o=0;a.start()}
;function Bl(){Wk.apply(this,arguments)}
y(Bl,Wk);function Cl(){Bl.g||(Bl.g=new Bl);return Bl.g}
Bl.prototype.Ha=function(a,b,c){void 0!==c&&Number.isNaN(Number(c))&&(c=void 0);var d=E("yt.scheduler.instance.addJob");return d?d(a,b,c):void 0===c?(a(),NaN):Vj(a,c||0)};
Bl.prototype.ba=function(a){if(void 0===a||!Number.isNaN(Number(a))){var b=E("yt.scheduler.instance.cancelJob");b?b(a):window.clearTimeout(a)}};
Bl.prototype.start=function(){var a=E("yt.scheduler.instance.start");a&&a()};
var Df=Cl();
S("web_scheduler_auto_init")&&!E("yt.scheduler.initialized")&&(F("yt.scheduler.instance.dispose",pl),F("yt.scheduler.instance.addJob",rl),F("yt.scheduler.instance.addImmediateJob",sl),F("yt.scheduler.instance.cancelJob",tl),F("yt.scheduler.instance.cancelAllJobs",ql),F("yt.scheduler.instance.start",vl),F("yt.scheduler.instance.pause",wl),F("yt.scheduler.instance.setPriorityThreshold",yl),F("yt.scheduler.instance.enablePriorityThreshold",zl),F("yt.scheduler.instance.clearPriorityThreshold",Al),F("yt.scheduler.initialized",
!0));function Dl(a){var b=new eh;(b=(b.h=dh(b.g))?a?new kh(b,a):b:null)||(a=new fh(a||"UserDataSharedStore"),b=a.g?a:null);this.g=(a=b)?new Zg(a):null;this.h=document.domain||window.location.hostname}
Dl.prototype.set=function(a,b,c,d){c=c||31104E3;this.remove(a);if(this.g)try{this.g.set(a,b,Date.now()+1E3*c);return}catch(f){}var e="";if(d)try{e=escape(Ag(b))}catch(f){return}else e=escape(b);b=this.h;Oe.set(""+a,e,{Qb:c,path:"/",domain:void 0===b?"youtube.com":b,secure:!1})};
Dl.prototype.get=function(a,b){var c=void 0,d=!this.g;if(!d)try{c=this.g.get(a)}catch(e){d=!0}if(d&&(c=Oe.get(""+a,void 0))&&(c=unescape(c),b))try{c=JSON.parse(c)}catch(e){this.remove(a),c=void 0}return c};
Dl.prototype.remove=function(a){this.g&&this.g.remove(a);var b=this.h;Oe.remove(""+a,"/",void 0===b?"youtube.com":b)};var El=function(){var a;return function(){a||(a=new Dl("ytidb"));return a}}();
function Fl(){var a;return null==(a=El())?void 0:a.get("LAST_RESULT_ENTRY_KEY",!0)}
;var Gl=[],Hl=!1;function Il(a){Hl||(Gl.push({type:"ERROR",payload:a}),10<Gl.length&&Gl.shift())}
function Jl(a,b){Hl||(Gl.push({type:"EVENT",eventType:a,payload:b}),10<Gl.length&&Gl.shift())}
;function Kl(a){if(0<=a.indexOf(":"))throw Error("Database name cannot contain ':'");}
function Ll(a){return a.substr(0,a.indexOf(":"))||a}
;var Ml=yc||Dc;var Nl={},Ol=(Nl.AUTH_INVALID="No user identifier specified.",Nl.EXPLICIT_ABORT="Transaction was explicitly aborted.",Nl.IDB_NOT_SUPPORTED="IndexedDB is not supported.",Nl.MISSING_INDEX="Index not created.",Nl.MISSING_OBJECT_STORES="Object stores not created.",Nl.DB_DELETED_BY_MISSING_OBJECT_STORES="Database is deleted because expected object stores were not created.",Nl.DB_REOPENED_BY_MISSING_OBJECT_STORES="Database is reopened because expected object stores were not created.",Nl.UNKNOWN_ABORT="Transaction was aborted for unknown reasons.",
Nl.QUOTA_EXCEEDED="The current transaction exceeded its quota limitations.",Nl.QUOTA_MAYBE_EXCEEDED="The current transaction may have failed because of exceeding quota limitations.",Nl.EXECUTE_TRANSACTION_ON_CLOSED_DB="Can't start a transaction on a closed database",Nl.INCOMPATIBLE_DB_VERSION="The binary is incompatible with the database version",Nl),Pl={},Ql=(Pl.AUTH_INVALID="ERROR",Pl.EXECUTE_TRANSACTION_ON_CLOSED_DB="WARNING",Pl.EXPLICIT_ABORT="IGNORED",Pl.IDB_NOT_SUPPORTED="ERROR",Pl.MISSING_INDEX=
"WARNING",Pl.MISSING_OBJECT_STORES="ERROR",Pl.DB_DELETED_BY_MISSING_OBJECT_STORES="WARNING",Pl.DB_REOPENED_BY_MISSING_OBJECT_STORES="WARNING",Pl.QUOTA_EXCEEDED="WARNING",Pl.QUOTA_MAYBE_EXCEEDED="WARNING",Pl.UNKNOWN_ABORT="WARNING",Pl.INCOMPATIBLE_DB_VERSION="WARNING",Pl),Rl={},Sl=(Rl.AUTH_INVALID=!1,Rl.EXECUTE_TRANSACTION_ON_CLOSED_DB=!1,Rl.EXPLICIT_ABORT=!1,Rl.IDB_NOT_SUPPORTED=!1,Rl.MISSING_INDEX=!1,Rl.MISSING_OBJECT_STORES=!1,Rl.DB_DELETED_BY_MISSING_OBJECT_STORES=!1,Rl.DB_REOPENED_BY_MISSING_OBJECT_STORES=
!1,Rl.QUOTA_EXCEEDED=!1,Rl.QUOTA_MAYBE_EXCEEDED=!0,Rl.UNKNOWN_ABORT=!0,Rl.INCOMPATIBLE_DB_VERSION=!1,Rl);function V(a,b,c,d,e){b=void 0===b?{}:b;c=void 0===c?Ol[a]:c;d=void 0===d?Ql[a]:d;e=void 0===e?Sl[a]:e;Tk.call(this,c,Object.assign({},{name:"YtIdbKnownError",isSw:void 0===self.document,isIframe:self!==self.top,type:a},b));this.type=a;this.message=c;this.level=d;this.g=e;Object.setPrototypeOf(this,V.prototype)}
y(V,Tk);function Tl(a,b){V.call(this,"MISSING_OBJECT_STORES",{expectedObjectStores:b,foundObjectStores:a},Ol.MISSING_OBJECT_STORES);Object.setPrototypeOf(this,Tl.prototype)}
y(Tl,V);function Ul(a,b){var c=Error.call(this);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.index=a;this.objectStore=b;Object.setPrototypeOf(this,Ul.prototype)}
y(Ul,Error);var Vl=["The database connection is closing","Can't start a transaction on a closed database","A mutation operation was attempted on a database that did not allow mutations"];
function Wl(a,b,c,d){b=Ll(b);var e=a instanceof Error?a:Error("Unexpected error: "+a);if(e instanceof V)return e;a={objectStoreNames:c,dbName:b,dbVersion:d};if("QuotaExceededError"===e.name)return new V("QUOTA_EXCEEDED",a);if(Ec&&"UnknownError"===e.name)return new V("QUOTA_MAYBE_EXCEEDED",a);if(e instanceof Ul)return new V("MISSING_INDEX",Object.assign({},a,{objectStore:e.objectStore,index:e.index}));if("InvalidStateError"===e.name&&Vl.some(function(f){return e.message.includes(f)}))return new V("EXECUTE_TRANSACTION_ON_CLOSED_DB",
a);
if("AbortError"===e.name)return new V("UNKNOWN_ABORT",a,e.message);e.args=[Object.assign({},a,{name:"IdbError",we:e.name})];e.level="WARNING";return e}
function Xl(a,b,c){var d=Fl();return new V("IDB_NOT_SUPPORTED",{context:{caller:a,publicName:b,version:c,hasSucceededOnce:null==d?void 0:d.hasSucceededOnce}})}
;function Yl(a){if(!a)throw Error();throw a;}
function Zl(a){return a}
function $l(a){this.g=a}
function am(a){function b(e){if("PENDING"===d.state.status){d.state={status:"REJECTED",reason:e};e=w(d.h);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
function c(e){if("PENDING"===d.state.status){d.state={status:"FULFILLED",value:e};e=w(d.g);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
var d=this;this.state={status:"PENDING"};this.g=[];this.h=[];a=a.g;try{a(c,b)}catch(e){b(e)}}
am.resolve=function(a){return new am(new $l(function(b,c){a instanceof am?a.then(b,c):b(a)}))};
am.reject=function(a){return new am(new $l(function(b,c){c(a)}))};
am.prototype.then=function(a,b){var c=this,d=null!=a?a:Zl,e=null!=b?b:Yl;return new am(new $l(function(f,g){"PENDING"===c.state.status?(c.g.push(function(){bm(c,c,d,f,g)}),c.h.push(function(){cm(c,c,e,f,g)})):"FULFILLED"===c.state.status?bm(c,c,d,f,g):"REJECTED"===c.state.status&&cm(c,c,e,f,g)}))};
function dm(a,b){a.then(void 0,b)}
function bm(a,b,c,d,e){try{if("FULFILLED"!==a.state.status)throw Error("calling handleResolve before the promise is fulfilled.");var f=c(a.state.value);f instanceof am?em(a,b,f,d,e):d(f)}catch(g){e(g)}}
function cm(a,b,c,d,e){try{if("REJECTED"!==a.state.status)throw Error("calling handleReject before the promise is rejected.");var f=c(a.state.reason);f instanceof am?em(a,b,f,d,e):d(f)}catch(g){e(g)}}
function em(a,b,c,d,e){b===c?e(new TypeError("Circular promise chain detected.")):c.then(function(f){f instanceof am?em(a,b,f,d,e):d(f)},function(f){e(f)})}
;function fm(a,b,c){function d(){c(a.error);f()}
function e(){b(a.result);f()}
function f(){try{a.removeEventListener("success",e),a.removeEventListener("error",d)}catch(g){}}
a.addEventListener("success",e);a.addEventListener("error",d)}
function gm(a){return new Promise(function(b,c){fm(a,b,c)})}
function hm(a){return new am(new $l(function(b,c){fm(a,b,c)}))}
;function im(a,b){return new am(new $l(function(c,d){function e(){var f=a?b(a):null;f?f.then(function(g){a=g;e()},d):c()}
e()}))}
;var jm=window,X=jm.ytcsi&&jm.ytcsi.now?jm.ytcsi.now:jm.performance&&jm.performance.timing&&jm.performance.now&&jm.performance.timing.navigationStart?function(){return jm.performance.timing.navigationStart+jm.performance.now()}:function(){return(new Date).getTime()};function km(a,b){this.g=a;this.options=b;this.transactionCount=0;this.i=Math.round(X());this.h=!1}
r=km.prototype;r.add=function(a,b,c){return lm(this,[a],{mode:"readwrite",W:!0},function(d){return d.objectStore(a).add(b,c)})};
r.clear=function(a){return lm(this,[a],{mode:"readwrite",W:!0},function(b){return b.objectStore(a).clear()})};
r.close=function(){this.g.close();var a;(null==(a=this.options)?0:a.closed)&&this.options.closed()};
function mm(a,b,c){a=a.g.createObjectStore(b,c);return new nm(a)}
r.delete=function(a,b){return lm(this,[a],{mode:"readwrite",W:!0},function(c){return c.objectStore(a).delete(b)})};
r.get=function(a,b){return lm(this,[a],{mode:"readonly",W:!0},function(c){return c.objectStore(a).get(b)})};
function om(a,b,c){return lm(a,[b],{mode:"readwrite",W:!0},function(d){d=d.objectStore(b);return hm(d.g.put(c,void 0))})}
r.objectStoreNames=function(){return Array.from(this.g.objectStoreNames)};
function lm(a,b,c,d){var e,f,g,h,k,l,n,m,q,p,t,u;return B(function(z){switch(z.g){case 1:var C={mode:"readonly",W:!1,tag:"IDB_TRANSACTION_TAG_UNKNOWN"};"string"===typeof c?C.mode=c:Object.assign(C,c);e=C;a.transactionCount++;f=e.W?3:1;g=0;case 2:if(h){z.B(4);break}g++;k=Math.round(X());va(z,5);l=a.g.transaction(b,e.mode);C=new pm(l);C=qm(C,d);return A(z,C,7);case 7:return n=z.h,m=Math.round(X()),rm(a,k,m,g,void 0,b.join(),e),z.return(n);case 5:q=wa(z);p=Math.round(X());t=Wl(q,a.g.name,b.join(),a.g.version);
if((u=t instanceof V&&!t.g)||g>=f)rm(a,k,p,g,t,b.join(),e),h=t;z.B(2);break;case 4:return z.return(Promise.reject(h))}})}
function rm(a,b,c,d,e,f,g){b=c-b;e?(e instanceof V&&("QUOTA_EXCEEDED"===e.type||"QUOTA_MAYBE_EXCEEDED"===e.type)&&Jl("QUOTA_EXCEEDED",{dbName:Ll(a.g.name),objectStoreNames:f,transactionCount:a.transactionCount,transactionMode:g.mode}),e instanceof V&&"UNKNOWN_ABORT"===e.type&&(c-=a.i,0>c&&c>=Math.pow(2,31)&&(c=0),Jl("TRANSACTION_UNEXPECTEDLY_ABORTED",{objectStoreNames:f,transactionDuration:b,transactionCount:a.transactionCount,dbDuration:c}),a.h=!0),sm(a,!1,d,f,b,g.tag),Il(e)):sm(a,!0,d,f,b,g.tag)}
function sm(a,b,c,d,e,f){Jl("TRANSACTION_ENDED",{objectStoreNames:d,connectionHasUnknownAbortedTransaction:a.h,duration:e,isSuccessful:b,tryCount:c,tag:void 0===f?"IDB_TRANSACTION_TAG_UNKNOWN":f})}
r.getName=function(){return this.g.name};
function nm(a){this.g=a}
r=nm.prototype;r.add=function(a,b){return hm(this.g.add(a,b))};
r.autoIncrement=function(){return this.g.autoIncrement};
r.clear=function(){return hm(this.g.clear()).then(function(){})};
function tm(a,b,c){a.g.createIndex(b,c,{unique:!1})}
function um(a,b){return vm(a,{query:b},function(c){return c.delete().then(function(){return c.continue()})}).then(function(){})}
r.delete=function(a){return a instanceof IDBKeyRange?um(this,a):hm(this.g.delete(a))};
r.get=function(a){return hm(this.g.get(a))};
r.index=function(a){try{return new wm(this.g.index(a))}catch(b){if(b instanceof Error&&"NotFoundError"===b.name)throw new Ul(a,this.g.name);throw b;}};
r.getName=function(){return this.g.name};
r.keyPath=function(){return this.g.keyPath};
function vm(a,b,c){a=a.g.openCursor(b.query,b.direction);return xm(a).then(function(d){return im(d,c)})}
function pm(a){var b=this;this.g=a;this.i=new Map;this.h=!1;this.done=new Promise(function(c,d){b.g.addEventListener("complete",function(){c()});
b.g.addEventListener("error",function(e){e.currentTarget===e.target&&d(b.g.error)});
b.g.addEventListener("abort",function(){var e=b.g.error;if(e)d(e);else if(!b.h){e=V;for(var f=b.g.objectStoreNames,g=[],h=0;h<f.length;h++){var k=f.item(h);if(null===k)throw Error("Invariant: item in DOMStringList is null");g.push(k)}e=new e("UNKNOWN_ABORT",{objectStoreNames:g.join(),dbName:b.g.db.name,mode:b.g.mode});d(e)}})})}
function qm(a,b){var c=new Promise(function(d,e){try{dm(b(a).then(function(f){d(f)}),e)}catch(f){e(f),a.abort()}});
return Promise.all([c,a.done]).then(function(d){return w(d).next().value})}
pm.prototype.abort=function(){this.g.abort();this.h=!0;throw new V("EXPLICIT_ABORT");};
pm.prototype.objectStore=function(a){a=this.g.objectStore(a);var b=this.i.get(a);b||(b=new nm(a),this.i.set(a,b));return b};
function wm(a){this.g=a}
r=wm.prototype;r.delete=function(a){return ym(this,{query:a},function(b){return b.delete().then(function(){return b.continue()})})};
r.get=function(a){return hm(this.g.get(a))};
r.getKey=function(a){return hm(this.g.getKey(a))};
r.keyPath=function(){return this.g.keyPath};
r.unique=function(){return this.g.unique};
function ym(a,b,c){a=a.g.openCursor(void 0===b.query?null:b.query,void 0===b.direction?"next":b.direction);return xm(a).then(function(d){return im(d,c)})}
function zm(a,b){this.request=a;this.cursor=b}
function xm(a){return hm(a).then(function(b){return b?new zm(a,b):null})}
r=zm.prototype;r.advance=function(a){this.cursor.advance(a);return xm(this.request)};
r.continue=function(a){this.cursor.continue(a);return xm(this.request)};
r.delete=function(){return hm(this.cursor.delete()).then(function(){})};
r.getKey=function(){return this.cursor.key};
r.ja=function(){return this.cursor.value};
r.update=function(a){return hm(this.cursor.update(a))};function Am(a,b,c){return new Promise(function(d,e){function f(){q||(q=new km(g.result,{closed:m}));return q}
var g=void 0!==b?self.indexedDB.open(a,b):self.indexedDB.open(a);var h=c.Tc,k=c.Uc,l=c.yd,n=c.upgrade,m=c.closed,q;g.addEventListener("upgradeneeded",function(p){try{if(null===p.newVersion)throw Error("Invariant: newVersion on IDbVersionChangeEvent is null");if(null===g.transaction)throw Error("Invariant: transaction on IDbOpenDbRequest is null");p.dataLoss&&"none"!==p.dataLoss&&Jl("IDB_DATA_CORRUPTED",{reason:p.dataLossMessage||"unknown reason",dbName:Ll(a)});var t=f(),u=new pm(g.transaction);n&&
n(t,function(z){return p.oldVersion<z&&p.newVersion>=z},u);
u.done.catch(function(z){e(z)})}catch(z){e(z)}});
g.addEventListener("success",function(){var p=g.result;k&&p.addEventListener("versionchange",function(){k(f())});
p.addEventListener("close",function(){Jl("IDB_UNEXPECTEDLY_CLOSED",{dbName:Ll(a),dbVersion:p.version});l&&l()});
d(f())});
g.addEventListener("error",function(){e(g.error)});
h&&g.addEventListener("blocked",function(){h()})})}
function Bm(a,b,c){c=void 0===c?{}:c;return Am(a,b,c)}
function Cm(a,b){b=void 0===b?{}:b;var c,d,e,f;return B(function(g){if(1==g.g)return va(g,2),c=self.indexedDB.deleteDatabase(a),d=b,(e=d.Tc)&&c.addEventListener("blocked",function(){e()}),A(g,gm(c),4);
if(2!=g.g)g.g=0,g.o=0;else throw f=wa(g),Wl(f,a,"",-1);})}
;function Dm(a,b){this.name=a;this.options=b;this.i=!0;this.o=this.l=0}
Dm.prototype.h=function(a,b,c){c=void 0===c?{}:c;return Bm(a,b,c)};
Dm.prototype.delete=function(a){a=void 0===a?{}:a;return Cm(this.name,a)};
function Em(a,b){return new V("INCOMPATIBLE_DB_VERSION",{dbName:a.name,oldVersion:a.options.version,newVersion:b})}
function Fm(a,b){if(!b)throw Xl("openWithToken",Ll(a.name));return Gm(a)}
function Gm(a){function b(){var f,g,h,k,l,n,m,q,p,t;return B(function(u){switch(u.g){case 1:return g=null!=(f=Error().stack)?f:"",va(u,2),A(u,a.h(a.name,a.options.version,d),4);case 4:for(var z=h=u.h,C=a.options,P=[],U=w(Object.keys(C.Za)),Y=U.next();!Y.done;Y=U.next()){Y=Y.value;var La=C.Za[Y],Ca=void 0===La.wd?Number.MAX_VALUE:La.wd;!(z.g.version>=La.hb)||z.g.version>=Ca||z.g.objectStoreNames.contains(Y)||P.push(Y)}k=P;if(0===k.length){u.B(5);break}l=Object.keys(a.options.Za);n=h.objectStoreNames();
if(a.o<T("ytidb_reopen_db_retries",0))return a.o++,h.close(),Il(new V("DB_REOPENED_BY_MISSING_OBJECT_STORES",{dbName:a.name,expectedObjectStores:l,foundObjectStores:n})),u.return(b());if(!(a.l<T("ytidb_remake_db_retries",1))){u.B(6);break}a.l++;return A(u,a.delete(),7);case 7:return Il(new V("DB_DELETED_BY_MISSING_OBJECT_STORES",{dbName:a.name,expectedObjectStores:l,foundObjectStores:n})),u.return(b());case 6:throw new Tl(n,l);case 5:return u.return(h);case 2:m=wa(u);if(m instanceof DOMException?
"VersionError"!==m.name:"DOMError"in self&&m instanceof DOMError?"VersionError"!==m.name:!(m instanceof Object&&"message"in m)||"An attempt was made to open a database using a lower version than the existing version."!==m.message){u.B(8);break}return A(u,a.h(a.name,void 0,Object.assign({},d,{upgrade:void 0})),9);case 9:q=u.h;p=q.g.version;if(void 0!==a.options.version&&p>a.options.version+1)throw q.close(),a.i=!1,Em(a,p);return u.return(q);case 8:throw c(),m instanceof Error&&!S("ytidb_async_stack_killswitch")&&
(m.stack=m.stack+"\n"+g.substring(g.indexOf("\n")+1)),Wl(m,a.name,"",null!=(t=a.options.version)?t:-1);}})}
function c(){a.g===e&&(a.g=void 0)}
if(!a.i)throw Em(a);if(a.g)return a.g;var d={Uc:function(f){f.close()},
closed:c,yd:c,upgrade:a.options.upgrade};var e=b();a.g=e;return a.g}
;var Hm=new Dm("YtIdbMeta",{Za:{databases:{hb:1}},upgrade:function(a,b){b(1)&&mm(a,"databases",{keyPath:"actualName"})}});
function Im(a,b){var c;return B(function(d){if(1==d.g)return A(d,Fm(Hm,b),2);c=d.h;return d.return(lm(c,["databases"],{W:!0,mode:"readwrite"},function(e){var f=e.objectStore("databases");return f.get(a.actualName).then(function(g){if(g?a.actualName!==g.actualName||a.publicName!==g.publicName||a.userIdentifier!==g.userIdentifier:1)return hm(f.g.put(a,void 0)).then(function(){})})}))})}
function Jm(a,b){var c;return B(function(d){if(1==d.g)return a?A(d,Fm(Hm,b),2):d.return();c=d.h;return d.return(c.delete("databases",a))})}
function Km(a,b){var c,d;return B(function(e){return 1==e.g?(c=[],A(e,Fm(Hm,b),2)):3!=e.g?(d=e.h,A(e,lm(d,["databases"],{W:!0,mode:"readonly"},function(f){c.length=0;return vm(f.objectStore("databases"),{},function(g){a(g.ja())&&c.push(g.ja());return g.continue()})}),3)):e.return(c)})}
function Lm(a){return Km(function(b){return"LogsDatabaseV2"===b.publicName&&void 0!==b.userIdentifier},a)}
;var Mm,Nm=new function(){}(new function(){});
function Om(){var a,b,c,d;return B(function(e){switch(e.g){case 1:a=Fl();if(null==(b=a)?0:b.hasSucceededOnce)return e.return(!0);var f;if(f=Ml)f=/WebKit\/([0-9]+)/.exec(Mb()),f=!!(f&&600<=parseInt(f[1],10));f&&(f=/WebKit\/([0-9]+)/.exec(Mb()),f=!(f&&602<=parseInt(f[1],10)));if(f||mc)return e.return(!1);try{if(c=self,!(c.indexedDB&&c.IDBIndex&&c.IDBKeyRange&&c.IDBObjectStore))return e.return(!1)}catch(g){return e.return(!1)}if(!("IDBTransaction"in self&&"objectStoreNames"in IDBTransaction.prototype))return e.return(!1);
va(e,2);d={actualName:"yt-idb-test-do-not-use",publicName:"yt-idb-test-do-not-use",userIdentifier:void 0};return A(e,Im(d,Nm),4);case 4:return A(e,Jm("yt-idb-test-do-not-use",Nm),5);case 5:return e.return(!0);case 2:return wa(e),e.return(!1)}})}
function Pm(){if(void 0!==Mm)return Mm;Hl=!0;return Mm=Om().then(function(a){Hl=!1;var b;if(null!=(b=El())&&b.g){var c;b={hasSucceededOnce:(null==(c=Fl())?void 0:c.hasSucceededOnce)||a};var d;null==(d=El())||d.set("LAST_RESULT_ENTRY_KEY",b,2592E3,!0)}return a})}
function Qm(){return E("ytglobal.idbToken_")||void 0}
function Rm(){var a=Qm();return a?Promise.resolve(a):Pm().then(function(b){(b=b?Nm:void 0)&&F("ytglobal.idbToken_",b);return b})}
;new Bg;function Sm(a){if(!Uk())throw a=new V("AUTH_INVALID",{dbName:a}),Il(a),a;var b=Vk();return{actualName:a+":"+b,publicName:a,userIdentifier:b}}
function Tm(a,b,c,d){var e,f,g,h,k,l;return B(function(n){switch(n.g){case 1:return f=null!=(e=Error().stack)?e:"",A(n,Rm(),2);case 2:g=n.h;if(!g)throw h=Xl("openDbImpl",a,b),S("ytidb_async_stack_killswitch")||(h.stack=h.stack+"\n"+f.substring(f.indexOf("\n")+1)),Il(h),h;Kl(a);k=c?{actualName:a,publicName:a,userIdentifier:void 0}:Sm(a);va(n,3);return A(n,Im(k,g),5);case 5:return A(n,Bm(k.actualName,b,d),6);case 6:return n.return(n.h);case 3:return l=wa(n),va(n,7),A(n,Jm(k.actualName,g),9);case 9:n.g=
8;n.o=0;break;case 7:wa(n);case 8:throw l;}})}
function Um(a,b,c){c=void 0===c?{}:c;return Tm(a,b,!1,c)}
function Vm(a,b,c){c=void 0===c?{}:c;return Tm(a,b,!0,c)}
function Wm(a,b){b=void 0===b?{}:b;var c,d;return B(function(e){if(1==e.g)return A(e,Rm(),2);if(3!=e.g){c=e.h;if(!c)return e.return();Kl(a);d=Sm(a);return A(e,Cm(d.actualName,b),3)}return A(e,Jm(d.actualName,c),0)})}
function Jn(a,b,c){a=a.map(function(d){return B(function(e){return 1==e.g?A(e,Cm(d.actualName,b),2):A(e,Jm(d.actualName,c),0)})});
return Promise.all(a).then(function(){})}
function Kn(){var a=void 0===a?{}:a;var b,c;return B(function(d){if(1==d.g)return A(d,Rm(),2);if(3!=d.g){b=d.h;if(!b)return d.return();Kl("LogsDatabaseV2");return A(d,Lm(b),3)}c=d.h;return A(d,Jn(c,a,b),0)})}
function Ln(a,b){b=void 0===b?{}:b;var c;return B(function(d){if(1==d.g)return A(d,Rm(),2);if(3!=d.g){c=d.h;if(!c)return d.return();Kl(a);return A(d,Cm(a,b),3)}return A(d,Jm(a,c),0)})}
;function Mn(a,b){Dm.call(this,a,b);this.options=b;Kl(a)}
y(Mn,Dm);function Nn(a,b){var c;return function(){c||(c=new Mn(a,b));return c}}
Mn.prototype.h=function(a,b,c){c=void 0===c?{}:c;return(this.options.Jb?Vm:Um)(a,b,Object.assign({},c))};
Mn.prototype.delete=function(a){a=void 0===a?{}:a;return(this.options.Jb?Ln:Wm)(this.name,a)};
function On(a,b){return Nn(a,b)}
;var Pn={},Qn=On("ytGcfConfig",{Za:(Pn.coldConfigStore={hb:1},Pn.hotConfigStore={hb:1},Pn),Jb:!1,upgrade:function(a,b){b(1)&&(tm(mm(a,"hotConfigStore",{keyPath:"key",autoIncrement:!0}),"hotTimestampIndex","timestamp"),tm(mm(a,"coldConfigStore",{keyPath:"key",autoIncrement:!0}),"coldTimestampIndex","timestamp"))},
version:1});function Rn(a){return Fm(Qn(),a)}
function Sn(a,b,c){var d,e,f;return B(function(g){switch(g.g){case 1:return d={config:a,hashData:b,timestamp:X()},A(g,Rn(c),2);case 2:return e=g.h,A(g,e.clear("hotConfigStore"),3);case 3:return A(g,om(e,"hotConfigStore",d),4);case 4:return f=g.h,g.return(f)}})}
function Tn(a,b,c,d){var e,f,g;return B(function(h){switch(h.g){case 1:return e={config:a,hashData:b,configData:c,timestamp:X()},A(h,Rn(d),2);case 2:return f=h.h,A(h,f.clear("coldConfigStore"),3);case 3:return A(h,om(f,"coldConfigStore",e),4);case 4:return g=h.h,h.return(g)}})}
function Un(a){var b,c;return B(function(d){return 1==d.g?A(d,Rn(a),2):3!=d.g?(b=d.h,c=void 0,A(d,lm(b,["coldConfigStore"],{mode:"readwrite",W:!0},function(e){return ym(e.objectStore("coldConfigStore").index("coldTimestampIndex"),{direction:"prev"},function(f){c=f.ja()})}),3)):d.return(c)})}
function Vn(a){var b,c;return B(function(d){return 1==d.g?A(d,Rn(a),2):3!=d.g?(b=d.h,c=void 0,A(d,lm(b,["hotConfigStore"],{mode:"readwrite",W:!0},function(e){return ym(e.objectStore("hotConfigStore").index("hotTimestampIndex"),{direction:"prev"},function(f){c=f.ja()})}),3)):d.return(c)})}
;function Wn(){We.call(this);this.h=[];this.g=[];var a=E("yt.gcf.config.hotUpdateCallbacks");a?(this.h=[].concat(x(a)),this.g=a):(this.g=[],F("yt.gcf.config.hotUpdateCallbacks",this.g))}
y(Wn,We);Wn.prototype.sa=function(){for(var a=w(this.h),b=a.next();!b.done;b=a.next()){var c=this.g;b=c.indexOf(b.value);0<=b&&c.splice(b,1)}this.h.length=0;We.prototype.sa.call(this)};function Xn(){this.h=0;this.i=new Wn}
function Yn(a,b,c){var d,e,f;return B(function(g){switch(g.g){case 1:if(!S("start_client_gcf")){g.B(0);break}c&&(a.l=c,F("yt.gcf.config.hotConfigGroup",a.l||null));a.g(b);d=Qm();if(!d){g.B(3);break}if(c){g.B(4);break}return A(g,Vn(d),5);case 5:e=g.h,c=null==(f=e)?void 0:f.config;case 4:return A(g,Sn(c,b,d),3);case 3:if(c)for(var h=c,k=w(a.i.g),l=k.next();!l.done;l=k.next())l=l.value,l(h);g.g=0}})}
function Zn(a,b,c){var d,e,f,g;return B(function(h){if(1==h.g){if(!S("start_client_gcf"))return h.B(0);a.coldHashData=b;F("yt.gcf.config.coldHashData",a.coldHashData||null);return(d=Qm())?c?h.B(4):A(h,Un(d),5):h.B(0)}4!=h.g&&(e=h.h,c=null==(f=e)?void 0:f.config);if(!c)return h.B(0);g=c.configData;return A(h,Tn(c,b,g,d),0)})}
Xn.prototype.g=function(a){this.hotHashData=a;F("yt.gcf.config.hotHashData",this.hotHashData||null)};function $n(){return"INNERTUBE_API_KEY"in Cj&&"INNERTUBE_API_VERSION"in Cj}
function ao(){return{gd:R("INNERTUBE_API_KEY"),hd:R("INNERTUBE_API_VERSION"),Mb:R("INNERTUBE_CONTEXT_CLIENT_CONFIG_INFO"),sc:R("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),jd:R("INNERTUBE_CONTEXT_CLIENT_NAME",1),uc:R("INNERTUBE_CONTEXT_CLIENT_VERSION"),wc:R("INNERTUBE_CONTEXT_HL"),vc:R("INNERTUBE_CONTEXT_GL"),kd:R("INNERTUBE_HOST_OVERRIDE")||"",md:!!R("INNERTUBE_USE_THIRD_PARTY_AUTH",!1),ld:!!R("INNERTUBE_OMIT_API_KEY_WHEN_AUTH_HEADER_IS_PRESENT",!1),appInstallData:R("SERIALIZED_CLIENT_CONFIG_DATA")}}
function bo(a){var b={client:{hl:a.wc,gl:a.vc,clientName:a.sc,clientVersion:a.uc,configInfo:a.Mb}};navigator.userAgent&&(b.client.userAgent=String(navigator.userAgent));var c=D.devicePixelRatio;c&&1!=c&&(b.client.screenDensityFloat=String(c));c=R("EXPERIMENTS_TOKEN","");""!==c&&(b.client.experimentsToken=c);c=Mj();0<c.length&&(b.request={internalExperimentFlags:c});co(a,void 0,b);eo(void 0,b);fo(void 0,b);go(a,void 0,b);ho(void 0,b);S("start_client_gcf")&&io(void 0,b);R("DELEGATED_SESSION_ID")&&!S("pageid_as_header_web")&&
(b.user={onBehalfOfUser:R("DELEGATED_SESSION_ID")});!S("fill_delegate_context_in_gel_killswitch")&&(a=R("INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT"))&&(b.user=Object.assign({},b.user,{serializedDelegationContext:a}));a=Object;c=a.assign;for(var d=b.client,e={},f=w(Object.entries(dk(R("DEVICE","")))),g=f.next();!g.done;g=f.next()){var h=w(g.value);g=h.next().value;h=h.next().value;"cbrand"===g?e.deviceMake=h:"cmodel"===g?e.deviceModel=h:"cbr"===g?e.browserName=h:"cbrver"===g?e.browserVersion=
h:"cos"===g?e.osName=h:"cosver"===g?e.osVersion=h:"cplatform"===g&&(e.platform=h)}b.client=c.call(a,d,e);return b}
function co(a,b,c){a=a.sc;if("WEB"===a||"MWEB"===a||1===a||2===a)if(b){c=Yd(b,Yi,96)||new Yi;var d=Mk();d=Object.keys(hj).indexOf(d);d=-1===d?null:d;null!==d&&fe(c,3,d);J(b,Yi,96,c)}else c&&(c.client.mainAppWebInfo=null!=(d=c.client.mainAppWebInfo)?d:{},c.client.mainAppWebInfo.webDisplayMode=Mk())}
function eo(a,b){var c=E("yt.embedded_player.embed_url");c&&(a?(b=Yd(a,dj,7)||new dj,K(b,4,c),J(a,dj,7,b)):b&&(b.thirdParty={embedUrl:c}))}
function fo(a,b){var c;if(S("web_log_memory_total_kbytes")&&(null==(c=D.navigator)?0:c.deviceMemory)){var d;c=null==(d=D.navigator)?void 0:d.deviceMemory;a?Sd(a,95,Ad(1E6*c)):b&&(b.client.memoryTotalKbytes=""+1E6*c)}}
function go(a,b,c){if(a.appInstallData)if(b){var d;c=null!=(d=Yd(b,Xi,62))?d:new Xi;K(c,6,a.appInstallData);J(b,Xi,62,c)}else c&&(c.client.configInfo=c.client.configInfo||{},c.client.configInfo.appInstallData=a.appInstallData)}
function ho(a,b){a:{var c=Sk();if(c){var d=Ok[c.type||"unknown"]||"CONN_UNKNOWN";c=Ok[c.effectiveType||"unknown"]||"CONN_UNKNOWN";"CONN_CELLULAR_UNKNOWN"===d&&"CONN_UNKNOWN"!==c&&(d=c);if("CONN_UNKNOWN"!==d)break a;if("CONN_UNKNOWN"!==c){d=c;break a}}d=void 0}d&&(a?fe(a,61,Pk[d]):b&&(b.client.connectionType=d));S("web_log_effective_connection_type")&&(d=Sk(),d=null!=d&&d.effectiveType?Rk.hasOwnProperty(d.effectiveType)?Rk[d.effectiveType]:"EFFECTIVE_CONNECTION_TYPE_UNKNOWN":void 0,d&&(a?fe(a,94,Qk[d]):
b&&(b.client.effectiveConnectionType=d)))}
function jo(a,b,c){c=void 0===c?{}:c;var d={};R("EOM_VISITOR_DATA")?d={"X-Goog-EOM-Visitor-Id":R("EOM_VISITOR_DATA")}:d={"X-Goog-Visitor-Id":c.visitorData||R("VISITOR_DATA","")};if(b&&b.includes("www.youtube-nocookie.com"))return d;b=c.oe||R("AUTHORIZATION");if(!b)if(a)b="Bearer "+E("gapi.auth.getToken")().ne;else{Lk.g||(Lk.g=new Lk);a={};if(c=Re([]))a.Authorization=c,c=void 0,void 0===c&&(c=Number(R("SESSION_INDEX",0)),c=isNaN(c)?0:c),S("voice_search_auth_header_removal")||(a["X-Goog-AuthUser"]=
c.toString()),"INNERTUBE_HOST_OVERRIDE"in Cj||(a["X-Origin"]=window.location.origin),"DELEGATED_SESSION_ID"in Cj&&(a["X-Goog-PageId"]=R("DELEGATED_SESSION_ID"));S("pageid_as_header_web")||delete a["X-Goog-PageId"];d=Object.assign({},d,a)}b&&(d.Authorization=b);return d}
function io(a,b){if(!Xn.g){var c=new Xn;Xn.g=c}c=Xn.g;var d=X()-c.h;if(0!==c.h&&d<T("send_config_hash_timer"))c=void 0;else{d=E("yt.gcf.config.coldConfigData");var e=E("yt.gcf.config.hotHashData"),f=E("yt.gcf.config.coldHashData");d&&e&&f&&(c.h=X());c={coldConfigData:d,hotHashData:e,coldHashData:f}}if(e=c)if(c=e.coldConfigData,d=e.coldHashData,e=e.hotHashData,c&&d&&e)if(a){var g;b=null!=(g=Yd(a,Xi,62))?g:new Xi;K(b,1,c);K(b,3,d);b.g(e);J(a,Xi,62,b)}else b&&(b.client.configInfo=b.client.configInfo||
{},b.client.configInfo.coldConfigData=c,b.client.configInfo.coldHashData=d,b.client.configInfo.hotHashData=e)}
;var ko="undefined"!==typeof TextEncoder?new TextEncoder:null,lo=ko?function(a){return ko.encode(a)}:function(a){for(var b=[],c=0,d=0;d<a.length;d++){var e=a.charCodeAt(d);
128>e?b[c++]=e:(2048>e?b[c++]=e>>6|192:(55296==(e&64512)&&d+1<a.length&&56320==(a.charCodeAt(d+1)&64512)?(e=65536+((e&1023)<<10)+(a.charCodeAt(++d)&1023),b[c++]=e>>18|240,b[c++]=e>>12&63|128):b[c++]=e>>12|224,b[c++]=e>>6&63|128),b[c++]=e&63|128)}a=new Uint8Array(b.length);for(c=0;c<a.length;c++)a[c]=b[c];return a};var mo=E("ytPubsub2Pubsub2Instance")||new M;M.prototype.subscribe=M.prototype.subscribe;M.prototype.unsubscribeByKey=M.prototype.rb;M.prototype.publish=M.prototype.ab;M.prototype.clear=M.prototype.clear;F("ytPubsub2Pubsub2Instance",mo);F("ytPubsub2Pubsub2SubscribedKeys",E("ytPubsub2Pubsub2SubscribedKeys")||{});F("ytPubsub2Pubsub2TopicToKeys",E("ytPubsub2Pubsub2TopicToKeys")||{});F("ytPubsub2Pubsub2IsAsync",E("ytPubsub2Pubsub2IsAsync")||{});F("ytPubsub2Pubsub2SkipSubKey",null);function no(a,b,c){c=void 0===c?{sampleRate:.1}:c;Math.random()<Math.min(.02,c.sampleRate/100)&&(a={Ce:a,Be:b},(b=E("ytPubsub2Pubsub2Instance"))&&b.publish.call(b,"meta_logging_csi_event".toString(),"meta_logging_csi_event",a))}
;var oo=void 0,po=void 0;function qo(){if(!po){var a=R("WORKER_SERIALIZATION_URL");if(a){if(a=a.privateDoNotAccessOrElseTrustedResourceUrlWrappedValue){var b=tb();a=b?b.createScriptURL(a):a;a=new xb(a,yb)}else a=null;po=a}else po=null}return po||void 0}
function ro(){var a=qo();if(!oo&&void 0!==a){var b=Worker;a instanceof xb&&a.constructor===xb?a=a.g:(Ka(a),a="type_error:TrustedResourceUrl");oo=new b(a,void 0)}return oo}
;var so=T("max_body_size_to_compress",5E5),to=T("min_body_size_to_compress",500),uo=!0,vo=0,wo=0,xo=T("compression_performance_threshold_lr",250),yo=T("slow_compressions_before_abandon_count",4),zo=!1,Ao=new Map,Bo=1,Co=!0;function Do(){if("function"===typeof Worker&&qo()&&!zo){var a=function(c){c=c.data;if("gzippedGelBatch"===c.op){var d=Ao.get(c.key);d&&(Eo(c.gzippedBatch,d.latencyPayload,d.url,d.options,d.sendFn),Ao.delete(c.key))}},b=ro();
b&&(b.addEventListener("message",a),b.onerror=function(){Ao.clear()},zo=!0)}}
function Fo(a,b,c,d,e){e=void 0===e?!1:e;var f={startTime:X(),ticks:{},infos:{}};if(uo)try{try{var g=(new Blob(b.split(""))).size}catch(n){Kj(n),g=null}if(null!=g&&(g>so||g<to))d(a,c);else{if(S("gzip_gel_with_worker")&&(S("initial_gzip_use_main_thread")&&!Co||!S("initial_gzip_use_main_thread"))){zo||Do();var h=ro();if(h&&!e){Ao.set(Bo,{latencyPayload:f,url:a,options:c,sendFn:d});h.postMessage({op:"gelBatchToGzip",serializedBatch:b,key:Bo});Bo++;return}}var k=lo(b);b=(b=void 0,{});b.ed=!0;var l=new Si(b);
l.push(k,!0);if(l.err)throw l.msg||wh[l.err];Eo(l.result,f,a,c,d)}}catch(n){Kj(n),d(a,c)}else d(a,c)}
function Eo(a,b,c,d,e){Co=!1;var f=X();b.ticks.gelc=f;wo++;S("disable_compression_due_to_performance_degredation")&&f-b.startTime>=xo&&(vo++,S("abandon_compression_after_N_slow_zips")?wo===T("compression_disable_point")&&vo>yo&&(uo=!1):uo=!1);S("gel_compression_csi_killswitch")||!S("log_gel_compression_latency")&&!S("log_gel_compression_latency_lr")||no("gel_compression",b,{sampleRate:.1});d.headers||(d.headers={});d.headers["Content-Encoding"]="gzip";d.postBody=a;d.postParams=void 0;e(c,d)}
;function Go(a){a=Object.assign({},a);delete a.Authorization;var b=Re();if(b){var c=new mg;c.update(R("INNERTUBE_API_KEY"));c.update(b);a.hash=Hc(c.digest(),3)}return a}
;var Ho;function Io(){Ho||(Ho=new Dl("yt.innertube"));return Ho}
function Jo(a,b,c,d){if(d)return null;d=Io().get("nextId",!0)||1;var e=Io().get("requests",!0)||{};e[d]={method:a,request:b,authState:Go(c),requestTime:Math.round(X())};Io().set("nextId",d+1,86400,!0);Io().set("requests",e,86400,!0);return d}
function Ko(a){var b=Io().get("requests",!0)||{};delete b[a];Io().set("requests",b,86400,!0)}
function Lo(a){var b=Io().get("requests",!0);if(b){for(var c in b){var d=b[c];if(!(6E4>Math.round(X())-d.requestTime)){var e=d.authState,f=Go(jo(!1));ob(e,f)&&(e=d.request,"requestTimeMs"in e&&(e.requestTimeMs=Math.round(X())),Mo(a,d.method,e,{}));delete b[c]}}Io().set("requests",b,86400,!0)}}
;function No(a){this.ub=this.g=!1;this.potentialEsfErrorCounter=this.h=0;this.handleError=function(){};
this.Sa=function(){};
this.now=Date.now;this.lb=!1;var b;this.Ic=null!=(b=a.Ic)?b:100;var c;this.Gc=null!=(c=a.Gc)?c:1;var d;this.Ec=null!=(d=a.Ec)?d:2592E6;var e;this.Dc=null!=(e=a.Dc)?e:12E4;var f;this.Fc=null!=(f=a.Fc)?f:5E3;var g;this.I=null!=(g=a.I)?g:void 0;this.zb=!!a.zb;var h;this.xb=null!=(h=a.xb)?h:.1;var k;this.Fb=null!=(k=a.Fb)?k:10;a.handleError&&(this.handleError=a.handleError);a.Sa&&(this.Sa=a.Sa);a.lb&&(this.lb=a.lb);a.ub&&(this.ub=a.ub);this.J=a.J;this.ha=a.ha;this.N=a.N;this.R=a.R;this.sendFn=a.sendFn;
this.Yb=a.Yb;this.Vb=a.Vb;Oo(this)&&(!this.J||this.J("networkless_logging"))&&Po(this)}
function Po(a){Oo(a)&&!a.lb&&(a.g=!0,a.zb&&Math.random()<=a.xb&&a.N.Vc(a.I),Qo(a),a.R.ca()&&a.qb(),a.R.Ka(a.Yb,a.qb.bind(a)),a.R.Ka(a.Vb,a.ic.bind(a)))}
r=No.prototype;r.writeThenSend=function(a,b){var c=this;b=void 0===b?{}:b;if(Oo(this)&&this.g){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.N.set(d,this.I).then(function(e){d.id=e;c.R.ca()&&Ro(c,d)}).catch(function(e){Ro(c,d);
So(c,e)})}else this.sendFn(a,b)};
r.sendThenWrite=function(a,b,c){var d=this;b=void 0===b?{}:b;if(Oo(this)&&this.g){var e={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.J&&this.J("nwl_skip_retry")&&(e.skipRetry=c);if(this.R.ca()||this.J&&this.J("nwl_aggressive_send_then_write")&&!e.skipRetry){if(!e.skipRetry){var f=b.onError?b.onError:function(){};
b.onError=function(g,h){return B(function(k){if(1==k.g)return A(k,d.N.set(e,d.I).catch(function(l){So(d,l)}),2);
f(g,h);k.g=0})}}this.sendFn(a,b,e.skipRetry)}else this.N.set(e,this.I).catch(function(g){d.sendFn(a,b,e.skipRetry);
So(d,g)})}else this.sendFn(a,b,this.J&&this.J("nwl_skip_retry")&&c)};
r.sendAndWrite=function(a,b){var c=this;b=void 0===b?{}:b;if(Oo(this)&&this.g){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0},e=!1,f=b.onSuccess?b.onSuccess:function(){};
d.options.onSuccess=function(g,h){void 0!==d.id?c.N.Ra(d.id,c.I):e=!0;c.R.Ma&&c.J&&c.J("vss_network_hint")&&c.R.Ma(!0);f(g,h)};
this.sendFn(d.url,d.options,void 0,!0);this.N.set(d,this.I).then(function(g){d.id=g;e&&c.N.Ra(d.id,c.I)}).catch(function(g){So(c,g)})}else this.sendFn(a,b,void 0,!0)};
r.qb=function(){var a=this;if(!Oo(this))throw Error("IndexedDB is not supported: throttleSend");this.h||(this.h=this.ha.qa(function(){var b;return B(function(c){if(1==c.g)return A(c,a.N.oc("NEW",a.I),2);if(3!=c.g)return b=c.h,b?A(c,Ro(a,b),3):(a.ic(),c.return());a.h&&(a.h=0,a.qb());c.g=0})},this.Ic))};
r.ic=function(){this.ha.ba(this.h);this.h=0};
function Ro(a,b){var c;return B(function(d){switch(d.g){case 1:if(!Oo(a))throw Error("IndexedDB is not supported: immediateSend");if(void 0===b.id){d.B(2);break}return A(d,a.N.od(b.id,a.I),3);case 3:(c=d.h)||a.Sa(Error("The request cannot be found in the database."));case 2:if(To(a,b,a.Ec)){d.B(4);break}a.Sa(Error("Networkless Logging: Stored logs request expired age limit"));if(void 0===b.id){d.B(5);break}return A(d,a.N.Ra(b.id,a.I),5);case 5:return d.return();case 4:b.skipRetry||(b=Uo(a,b));if(!b){d.B(0);
break}if(!b.skipRetry||void 0===b.id){d.B(8);break}return A(d,a.N.Ra(b.id,a.I),8);case 8:a.sendFn(b.url,b.options,!!b.skipRetry),d.g=0}})}
function Uo(a,b){if(!Oo(a))throw Error("IndexedDB is not supported: updateRequestHandlers");var c=b.options.onError?b.options.onError:function(){};
b.options.onError=function(e,f){var g,h,k,l;return B(function(n){switch(n.g){case 1:g=Vo(f);(h=Wo(f))&&a.J&&a.J("web_enable_error_204")&&a.handleError(Error("Request failed due to compression"),b.url,f);if(!(a.J&&a.J("nwl_consider_error_code")&&g||a.J&&!a.J("nwl_consider_error_code")&&a.potentialEsfErrorCounter<=a.Fb)){n.B(2);break}if(!a.R.Ib){n.B(3);break}return A(n,a.R.Ib(),3);case 3:if(a.R.ca()){n.B(2);break}c(e,f);if(!a.J||!a.J("nwl_consider_error_code")||void 0===(null==(k=b)?void 0:k.id)){n.B(6);
break}return A(n,a.N.Zb(b.id,a.I,!1),6);case 6:return n.return();case 2:if(a.J&&a.J("nwl_consider_error_code")&&!g&&a.potentialEsfErrorCounter>a.Fb)return n.return();a.potentialEsfErrorCounter++;if(void 0===(null==(l=b)?void 0:l.id)){n.B(8);break}return b.sendCount<a.Gc?A(n,a.N.Zb(b.id,a.I,!0,h?!1:void 0),12):A(n,a.N.Ra(b.id,a.I),8);case 12:a.ha.qa(function(){a.R.ca()&&a.qb()},a.Fc);
case 8:c(e,f),n.g=0}})};
var d=b.options.onSuccess?b.options.onSuccess:function(){};
b.options.onSuccess=function(e,f){var g;return B(function(h){if(1==h.g)return void 0===(null==(g=b)?void 0:g.id)?h.B(2):A(h,a.N.Ra(b.id,a.I),2);a.R.Ma&&a.J&&a.J("vss_network_hint")&&a.R.Ma(!0);d(e,f);h.g=0})};
return b}
function To(a,b,c){b=b.timestamp;return a.now()-b>=c?!1:!0}
function Qo(a){if(!Oo(a))throw Error("IndexedDB is not supported: retryQueuedRequests");a.N.oc("QUEUED",a.I).then(function(b){b&&!To(a,b,a.Dc)?a.ha.qa(function(){return B(function(c){if(1==c.g)return void 0===b.id?c.B(2):A(c,a.N.Zb(b.id,a.I),2);Qo(a);c.g=0})}):a.R.ca()&&a.qb()})}
function So(a,b){a.Kc&&!a.R.ca()?a.Kc(b):a.handleError(b)}
function Oo(a){return!!a.I||a.ub}
function Vo(a){var b;return(a=null==a?void 0:null==(b=a.error)?void 0:b.code)&&400<=a&&599>=a?!1:!0}
function Wo(a){var b;a=null==a?void 0:null==(b=a.error)?void 0:b.code;return!(400!==a&&415!==a)}
;var Xo;
function Yo(){if(Xo)return Xo();var a={};Xo=On("LogsDatabaseV2",{Za:(a.LogsRequestsStore={hb:2},a),Jb:!1,upgrade:function(b,c,d){c(2)&&mm(b,"LogsRequestsStore",{keyPath:"id",autoIncrement:!0});c(3);c(5)&&(d=d.objectStore("LogsRequestsStore"),d.g.indexNames.contains("newRequest")&&d.g.deleteIndex("newRequest"),tm(d,"newRequestV2",["status","interface","timestamp"]));c(7)&&b.g.objectStoreNames.contains("sapisid")&&b.g.deleteObjectStore("sapisid");c(9)&&b.g.objectStoreNames.contains("SWHealthLog")&&b.g.deleteObjectStore("SWHealthLog")},
version:9});return Xo()}
;function Zo(a){return Fm(Yo(),a)}
function $o(a,b){var c,d,e,f;return B(function(g){if(1==g.g)return c={startTime:X(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_WRITE"},ticks:{}},A(g,Zo(b),2);if(3!=g.g)return d=g.h,e=Object.assign({},a,{options:JSON.parse(JSON.stringify(a.options)),interface:R("INNERTUBE_CONTEXT_CLIENT_NAME",0)}),A(g,om(d,"LogsRequestsStore",e),3);f=g.h;c.ticks.tc=X();ap(c);return g.return(f)})}
function bp(a,b){var c,d,e,f,g,h,k;return B(function(l){if(1==l.g)return c={startTime:X(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_READ"},ticks:{}},A(l,Zo(b),2);if(3!=l.g)return d=l.h,e=R("INNERTUBE_CONTEXT_CLIENT_NAME",0),f=[a,e,0],g=[a,e,X()],h=IDBKeyRange.bound(f,g),k=void 0,A(l,lm(d,["LogsRequestsStore"],{mode:"readwrite",W:!0},function(n){return ym(n.objectStore("LogsRequestsStore").index("newRequestV2"),{query:h,direction:"prev"},function(m){m.ja()&&(k=m.ja(),"NEW"===a&&(k.status="QUEUED",
m.update(k)))})}),3);
c.ticks.tc=X();ap(c);return l.return(k)})}
function cp(a,b){var c;return B(function(d){if(1==d.g)return A(d,Zo(b),2);c=d.h;return d.return(lm(c,["LogsRequestsStore"],{mode:"readwrite",W:!0},function(e){var f=e.objectStore("LogsRequestsStore");return f.get(a).then(function(g){if(g)return g.status="QUEUED",hm(f.g.put(g,void 0)).then(function(){return g})})}))})}
function dp(a,b,c,d){c=void 0===c?!0:c;var e;return B(function(f){if(1==f.g)return A(f,Zo(b),2);e=f.h;return f.return(lm(e,["LogsRequestsStore"],{mode:"readwrite",W:!0},function(g){var h=g.objectStore("LogsRequestsStore");return h.get(a).then(function(k){return k?(k.status="NEW",c&&(k.sendCount+=1),void 0!==d&&(k.options.compress=d),hm(h.g.put(k,void 0)).then(function(){return k})):am.resolve(void 0)})}))})}
function ep(a,b){var c;return B(function(d){if(1==d.g)return A(d,Zo(b),2);c=d.h;return d.return(c.delete("LogsRequestsStore",a))})}
function fp(a){var b,c;return B(function(d){if(1==d.g)return A(d,Zo(a),2);b=d.h;c=X()-2592E6;return A(d,lm(b,["LogsRequestsStore"],{mode:"readwrite",W:!0},function(e){return vm(e.objectStore("LogsRequestsStore"),{},function(f){if(f.ja().timestamp<=c)return f.delete().then(function(){return f.continue()})})}),0)})}
function gp(){B(function(a){return A(a,Kn(),0)})}
function ap(a){S("nwl_csi_killswitch")||no("networkless_performance",a,{sampleRate:1})}
;var hp={accountStateChangeSignedIn:23,accountStateChangeSignedOut:24,delayedEventMetricCaptured:11,latencyActionBaselined:6,latencyActionInfo:7,latencyActionTicked:5,offlineTransferStatusChanged:2,offlineImageDownload:335,playbackStartStateChanged:9,systemHealthCaptured:3,mangoOnboardingCompleted:10,mangoPushNotificationReceived:230,mangoUnforkDbMigrationError:121,mangoUnforkDbMigrationSummary:122,mangoUnforkDbMigrationPreunforkDbVersionNumber:133,mangoUnforkDbMigrationPhoneMetadata:134,mangoUnforkDbMigrationPhoneStorage:135,
mangoUnforkDbMigrationStep:142,mangoAsyncApiMigrationEvent:223,mangoDownloadVideoResult:224,mangoHomepageVideoCount:279,mangoHomeV3State:295,mangoImageClientCacheHitEvent:273,sdCardStatusChanged:98,framesDropped:12,thumbnailHovered:13,deviceRetentionInfoCaptured:14,thumbnailLoaded:15,backToAppEvent:318,streamingStatsCaptured:17,offlineVideoShared:19,appCrashed:20,youThere:21,offlineStateSnapshot:22,mdxSessionStarted:25,mdxSessionConnected:26,mdxSessionDisconnected:27,bedrockResourceConsumptionSnapshot:28,
nextGenWatchWatchSwiped:29,kidsAccountsSnapshot:30,zeroStepChannelCreated:31,tvhtml5SearchCompleted:32,offlineSharePairing:34,offlineShareUnlock:35,mdxRouteDistributionSnapshot:36,bedrockRepetitiveActionTimed:37,unpluggedDegradationInfo:229,uploadMp4HeaderMoved:38,uploadVideoTranscoded:39,uploadProcessorStarted:46,uploadProcessorEnded:47,uploadProcessorReady:94,uploadProcessorRequirementPending:95,uploadProcessorInterrupted:96,uploadFrontendEvent:241,assetPackDownloadStarted:41,assetPackDownloaded:42,
assetPackApplied:43,assetPackDeleted:44,appInstallAttributionEvent:459,playbackSessionStopped:45,adBlockerMessagingShown:48,distributionChannelCaptured:49,dataPlanCpidRequested:51,detailedNetworkTypeCaptured:52,sendStateUpdated:53,receiveStateUpdated:54,sendDebugStateUpdated:55,receiveDebugStateUpdated:56,kidsErrored:57,mdxMsnSessionStatsFinished:58,appSettingsCaptured:59,mdxWebSocketServerHttpError:60,mdxWebSocketServer:61,startupCrashesDetected:62,coldStartInfo:435,offlinePlaybackStarted:63,liveChatMessageSent:225,
liveChatUserPresent:434,liveChatBeingModerated:457,liveCreationCameraUpdated:64,liveCreationEncodingCaptured:65,liveCreationError:66,liveCreationHealthUpdated:67,liveCreationVideoEffectsCaptured:68,liveCreationStageOccured:75,liveCreationBroadcastScheduled:123,liveCreationArchiveReplacement:149,liveCreationCostreamingConnection:421,liveCreationStreamWebrtcStats:288,mdxSessionRecoveryStarted:69,mdxSessionRecoveryCompleted:70,mdxSessionRecoveryStopped:71,visualElementShown:72,visualElementHidden:73,
visualElementGestured:78,visualElementStateChanged:208,screenCreated:156,playbackAssociated:202,visualElementAttached:215,playbackContextEvent:214,cloudCastingPlaybackStarted:74,webPlayerApiCalled:76,tvhtml5AccountDialogOpened:79,foregroundHeartbeat:80,foregroundHeartbeatScreenAssociated:111,kidsOfflineSnapshot:81,mdxEncryptionSessionStatsFinished:82,playerRequestCompleted:83,liteSchedulerStatistics:84,mdxSignIn:85,spacecastMetadataLookupRequested:86,spacecastBatchLookupRequested:87,spacecastSummaryRequested:88,
spacecastPlayback:89,spacecastDiscovery:90,tvhtml5LaunchUrlComponentChanged:91,mdxBackgroundPlaybackRequestCompleted:92,mdxBrokenAdditionalDataDeviceDetected:93,tvhtml5LocalStorage:97,tvhtml5DeviceStorageStatus:147,autoCaptionsAvailable:99,playbackScrubbingEvent:339,flexyState:100,interfaceOrientationCaptured:101,mainAppBrowseFragmentCache:102,offlineCacheVerificationFailure:103,offlinePlaybackExceptionDigest:217,vrCopresenceStats:104,vrCopresenceSyncStats:130,vrCopresenceCommsStats:137,vrCopresencePartyStats:153,
vrCopresenceEmojiStats:213,vrCopresenceEvent:141,vrCopresenceFlowTransitEvent:160,vrCowatchPartyEvent:492,vrPlaybackEvent:345,kidsAgeGateTracking:105,offlineDelayAllowedTracking:106,mainAppAutoOfflineState:107,videoAsThumbnailDownload:108,videoAsThumbnailPlayback:109,liteShowMore:110,renderingError:118,kidsProfilePinGateTracking:119,abrTrajectory:124,scrollEvent:125,streamzIncremented:126,kidsProfileSwitcherTracking:127,kidsProfileCreationTracking:129,buyFlowStarted:136,mbsConnectionInitiated:138,
mbsPlaybackInitiated:139,mbsLoadChildren:140,liteProfileFetcher:144,mdxRemoteTransaction:146,reelPlaybackError:148,reachabilityDetectionEvent:150,mobilePlaybackEvent:151,courtsidePlayerStateChanged:152,musicPersistentCacheChecked:154,musicPersistentCacheCleared:155,playbackInterrupted:157,playbackInterruptionResolved:158,fixFopFlow:159,anrDetection:161,backstagePostCreationFlowEnded:162,clientError:163,gamingAccountLinkStatusChanged:164,liteHousewarming:165,buyFlowEvent:167,kidsParentalGateTracking:168,
kidsSignedOutSettingsStatus:437,kidsSignedOutPauseHistoryFixStatus:438,tvhtml5WatchdogViolation:444,ypcUpgradeFlow:169,yongleStudy:170,ypcUpdateFlowStarted:171,ypcUpdateFlowCancelled:172,ypcUpdateFlowSucceeded:173,ypcUpdateFlowFailed:174,liteGrowthkitPromo:175,paymentFlowStarted:341,transactionFlowShowPaymentDialog:405,transactionFlowStarted:176,transactionFlowSecondaryDeviceStarted:222,transactionFlowSecondaryDeviceSignedOutStarted:383,transactionFlowCancelled:177,transactionFlowPaymentCallBackReceived:387,
transactionFlowPaymentSubmitted:460,transactionFlowPaymentSucceeded:329,transactionFlowSucceeded:178,transactionFlowFailed:179,transactionFlowPlayBillingConnectionStartEvent:428,transactionFlowSecondaryDeviceSuccess:458,transactionFlowErrorEvent:411,liteVideoQualityChanged:180,watchBreakEnablementSettingEvent:181,watchBreakFrequencySettingEvent:182,videoEffectsCameraPerformanceMetrics:183,adNotify:184,startupTelemetry:185,playbackOfflineFallbackUsed:186,outOfMemory:187,ypcPauseFlowStarted:188,ypcPauseFlowCancelled:189,
ypcPauseFlowSucceeded:190,ypcPauseFlowFailed:191,uploadFileSelected:192,ypcResumeFlowStarted:193,ypcResumeFlowCancelled:194,ypcResumeFlowSucceeded:195,ypcResumeFlowFailed:196,adsClientStateChange:197,ypcCancelFlowStarted:198,ypcCancelFlowCancelled:199,ypcCancelFlowSucceeded:200,ypcCancelFlowFailed:201,ypcCancelFlowGoToPaymentProcessor:402,ypcDeactivateFlowStarted:320,ypcRedeemFlowStarted:203,ypcRedeemFlowCancelled:204,ypcRedeemFlowSucceeded:205,ypcRedeemFlowFailed:206,ypcFamilyCreateFlowStarted:258,
ypcFamilyCreateFlowCancelled:259,ypcFamilyCreateFlowSucceeded:260,ypcFamilyCreateFlowFailed:261,ypcFamilyManageFlowStarted:262,ypcFamilyManageFlowCancelled:263,ypcFamilyManageFlowSucceeded:264,ypcFamilyManageFlowFailed:265,restoreContextEvent:207,embedsAdEvent:327,autoplayTriggered:209,clientDataErrorEvent:210,experimentalVssValidation:211,tvhtml5TriggeredEvent:212,tvhtml5FrameworksFieldTrialResult:216,tvhtml5FrameworksFieldTrialStart:220,musicOfflinePreferences:218,watchTimeSegment:219,appWidthLayoutError:221,
accountRegistryChange:226,userMentionAutoCompleteBoxEvent:227,downloadRecommendationEnablementSettingEvent:228,musicPlaybackContentModeChangeEvent:231,offlineDbOpenCompleted:232,kidsFlowEvent:233,kidsFlowCorpusSelectedEvent:234,videoEffectsEvent:235,unpluggedOpsEogAnalyticsEvent:236,playbackAudioRouteEvent:237,interactionLoggingDebugModeError:238,offlineYtbRefreshed:239,kidsFlowError:240,musicAutoplayOnLaunchAttempted:242,deviceContextActivityEvent:243,deviceContextEvent:244,templateResolutionException:245,
musicSideloadedPlaylistServiceCalled:246,embedsStorageAccessNotChecked:247,embedsHasStorageAccessResult:248,embedsItpPlayedOnReload:249,embedsRequestStorageAccessResult:250,embedsShouldRequestStorageAccessResult:251,embedsRequestStorageAccessState:256,embedsRequestStorageAccessFailedState:257,embedsItpWatchLaterResult:266,searchSuggestDecodingPayloadFailure:252,siriShortcutActivated:253,tvhtml5KeyboardPerformance:254,latencyActionSpan:255,elementsLog:267,ytbFileOpened:268,tfliteModelError:269,apiTest:270,
yongleUsbSetup:271,touStrikeInterstitialEvent:272,liteStreamToSave:274,appBundleClientEvent:275,ytbFileCreationFailed:276,adNotifyFailure:278,ytbTransferFailed:280,blockingRequestFailed:281,liteAccountSelector:282,liteAccountUiCallbacks:283,dummyPayload:284,browseResponseValidationEvent:285,entitiesError:286,musicIosBackgroundFetch:287,mdxNotificationEvent:289,layersValidationError:290,musicPwaInstalled:291,liteAccountCleanup:292,html5PlayerHealthEvent:293,watchRestoreAttempt:294,liteAccountSignIn:296,
notaireEvent:298,kidsVoiceSearchEvent:299,adNotifyFilled:300,delayedEventDropped:301,analyticsSearchEvent:302,systemDarkThemeOptOutEvent:303,flowEvent:304,networkConnectivityBaselineEvent:305,ytbFileImported:306,downloadStreamUrlExpired:307,directSignInEvent:308,lyricImpressionEvent:309,accessibilityStateEvent:310,tokenRefreshEvent:311,genericAttestationExecution:312,tvhtml5VideoSeek:313,unpluggedAutoPause:314,scrubbingEvent:315,bedtimeReminderEvent:317,tvhtml5UnexpectedRestart:319,tvhtml5StabilityTraceEvent:478,
tvhtml5OperationHealth:467,tvhtml5WatchKeyEvent:321,voiceLanguageChanged:322,tvhtml5LiveChatStatus:323,parentToolsCorpusSelectedEvent:324,offerAdsEnrollmentInitiated:325,networkQualityIntervalEvent:326,deviceStartupMetrics:328,heartbeatActionPlayerTransitioned:330,tvhtml5Lifecycle:331,heartbeatActionPlayerHalted:332,adaptiveInlineMutedSettingEvent:333,mainAppLibraryLoadingState:334,thirdPartyLogMonitoringEvent:336,appShellAssetLoadReport:337,tvhtml5AndroidAttestation:338,tvhtml5StartupSoundEvent:340,
iosBackgroundRefreshTask:342,iosBackgroundProcessingTask:343,sliEventBatch:344,postImpressionEvent:346,musicSideloadedPlaylistExport:347,idbUnexpectedlyClosed:348,voiceSearchEvent:349,mdxSessionCastEvent:350,idbQuotaExceeded:351,idbTransactionEnded:352,idbTransactionAborted:353,tvhtml5KeyboardLogging:354,idbIsSupportedCompleted:355,creatorStudioMobileEvent:356,idbDataCorrupted:357,parentToolsAppChosenEvent:358,webViewBottomSheetResized:359,activeStateControllerScrollPerformanceSummary:360,navigatorValidation:361,
mdxSessionHeartbeat:362,clientHintsPolyfillDiagnostics:363,clientHintsPolyfillEvent:364,proofOfOriginTokenError:365,kidsAddedAccountSummary:366,musicWearableDevice:367,ypcRefundFlowEvent:368,tvhtml5PlaybackMeasurementEvent:369,tvhtml5WatermarkMeasurementEvent:370,clientExpGcfPropagationEvent:371,mainAppReferrerIntent:372,leaderLockEnded:373,leaderLockAcquired:374,googleHatsEvent:375,persistentLensLaunchEvent:376,parentToolsChildWelcomeChosenEvent:378,browseThumbnailPreloadEvent:379,finalPayload:380,
mdxDialAdditionalDataUpdateEvent:381,webOrchestrationTaskLifecycleRecord:382,startupSignalEvent:384,accountError:385,gmsDeviceCheckEvent:386,accountSelectorEvent:388,accountUiCallbacks:389,mdxDialAdditionalDataProbeEvent:390,downloadsSearchIcingApiStats:391,downloadsSearchIndexUpdatedEvent:397,downloadsSearchIndexSnapshot:398,dataPushClientEvent:392,kidsCategorySelectedEvent:393,mdxDeviceManagementSnapshotEvent:394,prefetchRequested:395,prefetchableCommandExecuted:396,gelDebuggingEvent:399,webLinkTtsPlayEnd:400,
clipViewInvalid:401,persistentStorageStateChecked:403,cacheWipeoutEvent:404,playerEvent:410,sfvEffectPipelineStartedEvent:412,sfvEffectPipelinePausedEvent:429,sfvEffectPipelineEndedEvent:413,sfvEffectChosenEvent:414,sfvEffectLoadedEvent:415,sfvEffectUserInteractionEvent:465,sfvEffectFirstFrameProcessedLatencyEvent:416,sfvEffectAggregatedFramesProcessedLatencyEvent:417,sfvEffectAggregatedFramesDroppedEvent:418,sfvEffectPipelineErrorEvent:430,sfvEffectGraphFrozenEvent:419,sfvEffectGlThreadBlockedEvent:420,
mdeVideoChangedEvent:442,mdePlayerPerformanceMetrics:472,genericClientExperimentEvent:423,homePreloadTaskScheduled:424,homePreloadTaskExecuted:425,homePreloadCacheHit:426,polymerPropertyChangedInObserver:427,applicationStarted:431,networkCronetRttBatch:432,networkCronetRttSummary:433,repeatChapterLoopEvent:436,seekCancellationEvent:462,lockModeTimeoutEvent:483,offlineTransferStarted:4,musicOfflineMixtapePreferencesChanged:16,mangoDailyNewVideosNotificationAttempt:40,mangoDailyNewVideosNotificationError:77,
dtwsPlaybackStarted:112,dtwsTileFetchStarted:113,dtwsTileFetchCompleted:114,dtwsTileFetchStatusChanged:145,dtwsKeyframeDecoderBufferSent:115,dtwsTileUnderflowedOnNonkeyframe:116,dtwsBackfillFetchStatusChanged:143,dtwsBackfillUnderflowed:117,dtwsAdaptiveLevelChanged:128,blockingVisitorIdTimeout:277,liteSocial:18,mobileJsInvocation:297,biscottiBasedDetection:439,coWatchStateChange:440,embedsVideoDataDidChange:441,shortsFirst:443,cruiseControlEvent:445,qoeClientLoggingContext:446,atvRecommendationJobExecuted:447,
tvhtml5UserFeedback:448,producerProjectCreated:449,producerProjectOpened:450,producerProjectDeleted:451,producerProjectElementAdded:453,producerProjectElementRemoved:454,tvhtml5ShowClockEvent:455,deviceCapabilityCheckMetrics:456,youtubeClearcutEvent:461,offlineBrowseFallbackEvent:463,getCtvTokenEvent:464,startupDroppedFramesSummary:466,screenshotEvent:468,miniAppPlayEvent:469,elementsDebugCounters:470,fontLoadEvent:471,webKillswitchReceived:473,webKillswitchExecuted:474,cameraOpenEvent:475,manualSmoothnessMeasurement:476,
tvhtml5AppQualityEvent:477,polymerPropertyAccessEvent:479,miniAppSdkUsage:480,cobaltTelemetryEvent:481,crossDevicePlayback:482,channelCreatedWithObakeImage:484,channelEditedWithObakeImage:485,offlineDeleteEvent:486,crossDeviceNotificationTransfer:487,androidIntentEvent:488,unpluggedAmbientInterludesCounterfactualEvent:489,keyPlaysPlayback:490,shortsCreationFallbackEvent:493,vssData:491,castMatch:494,miniAppPerformanceMetrics:495};var ip={},jp=On("ServiceWorkerLogsDatabase",{Za:(ip.SWHealthLog={hb:1},ip),Jb:!0,upgrade:function(a,b){b(1)&&tm(mm(a,"SWHealthLog",{keyPath:"id",autoIncrement:!0}),"swHealthNewRequest",["interface","timestamp"])},
version:1});function kp(a){return Fm(jp(),a)}
function lp(a){var b,c;B(function(d){if(1==d.g)return A(d,kp(a),2);b=d.h;c=X()-2592E6;return A(d,lm(b,["SWHealthLog"],{mode:"readwrite",W:!0},function(e){return vm(e.objectStore("SWHealthLog"),{},function(f){if(f.ja().timestamp<=c)return f.delete().then(function(){return f.continue()})})}),0)})}
function mp(a){var b;return B(function(c){if(1==c.g)return A(c,kp(a),2);b=c.h;return A(c,b.clear("SWHealthLog"),0)})}
;var np={},op=0;function pp(a){var b=new Image,c=""+op++;np[c]=b;b.onload=b.onerror=function(){delete np[c]};
b.src=a}
;function qp(){this.g=new Map;this.h=!1}
function rp(){if(!qp.g){var a=E("yt.networkRequestMonitor.instance")||new qp;F("yt.networkRequestMonitor.instance",a);qp.g=a}return qp.g}
qp.prototype.requestComplete=function(a,b){b&&(this.h=!0);a=this.removeParams(a);this.g.get(a)||this.g.set(a,b)};
qp.prototype.isEndpointCFR=function(a){a=this.removeParams(a);return(a=this.g.get(a))?!1:!1===a&&this.h?!0:null};
qp.prototype.removeParams=function(a){return a.split("?")[0]};
qp.prototype.removeParams=qp.prototype.removeParams;qp.prototype.isEndpointCFR=qp.prototype.isEndpointCFR;qp.prototype.requestComplete=qp.prototype.requestComplete;qp.getInstance=rp;var sp;function tp(){sp||(sp=new Dl("yt.offline"));return sp}
function up(a){if(S("offline_error_handling")){var b=tp().get("errors",!0)||{};b[a.message]={name:a.name,stack:a.stack};a.level&&(b[a.message].level=a.level);tp().set("errors",b,2592E3,!0)}}
;function Z(){wf.call(this);var a=this;this.i=!1;this.h=Cf();this.h.Ka("networkstatus-online",function(){if(a.i&&S("offline_error_handling")){var b=tp().get("errors",!0);if(b){for(var c in b)if(b[c]){var d=new Tk(c,"sent via offline_errors");d.name=b[c].name;d.stack=b[c].stack;d.level=b[c].level;Jj(d)}tp().set("errors",{},2592E3,!0)}}})}
y(Z,wf);function vp(){if(!Z.g){var a=E("yt.networkStatusManager.instance")||new Z;F("yt.networkStatusManager.instance",a);Z.g=a}return Z.g}
r=Z.prototype;r.ca=function(){return this.h.ca()};
r.Ma=function(a){this.h.h=a};
r.cd=function(){var a=window.navigator.onLine;return void 0===a?!0:a};
r.Yc=function(){this.i=!0};
r.Ka=function(a,b){return this.h.Ka(a,b)};
r.Ib=function(a){a=Af(this.h,a);a.then(function(b){S("use_cfr_monitor")&&rp().requestComplete("generate_204",b)});
return a};
Z.prototype.sendNetworkCheckRequest=Z.prototype.Ib;Z.prototype.listen=Z.prototype.Ka;Z.prototype.enableErrorFlushing=Z.prototype.Yc;Z.prototype.getWindowStatus=Z.prototype.cd;Z.prototype.networkStatusHint=Z.prototype.Ma;Z.prototype.isNetworkAvailable=Z.prototype.ca;Z.getInstance=vp;function wp(a){a=void 0===a?{}:a;wf.call(this);var b=this;this.h=this.u=0;this.i=vp();var c=E("yt.networkStatusManager.instance.listen").bind(this.i);c&&(a.Hb?(this.Hb=a.Hb,c("networkstatus-online",function(){xp(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){xp(b,"publicytnetworkstatus-offline")})):(c("networkstatus-online",function(){xf(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){xf(b,"publicytnetworkstatus-offline")})))}
y(wp,wf);wp.prototype.ca=function(){var a=E("yt.networkStatusManager.instance.isNetworkAvailable");return a?a.bind(this.i)():!0};
wp.prototype.Ma=function(a){var b=E("yt.networkStatusManager.instance.networkStatusHint").bind(this.i);b&&b(a)};
wp.prototype.Ib=function(a){var b=this,c;return B(function(d){c=E("yt.networkStatusManager.instance.sendNetworkCheckRequest").bind(b.i);return S("skip_network_check_if_cfr")&&rp().isEndpointCFR("generate_204")?d.return(new Promise(function(e){var f;b.Ma((null==(f=window.navigator)?void 0:f.onLine)||!0);e(b.ca())})):c?d.return(c(a)):d.return(!0)})};
function xp(a,b){a.Hb?a.h?(Df.ba(a.u),a.u=Df.qa(function(){a.o!==b&&(xf(a,b),a.o=b,a.h=X())},a.Hb-(X()-a.h))):(xf(a,b),a.o=b,a.h=X()):xf(a,b)}
;var yp;function zp(){var a=No.call;yp||(yp=new wp({se:!0,re:!0}));a.call(No,this,{N:{Vc:fp,Ra:ep,oc:bp,od:cp,Zb:dp,set:$o},R:yp,handleError:function(b,c,d){var e,f=null==d?void 0:null==(e=d.error)?void 0:e.code;if(400===f||415===f){var g;Kj(new Tk(b.message,c,null==d?void 0:null==(g=d.error)?void 0:g.code),void 0,void 0,void 0,!0)}else Jj(b)},
Sa:Kj,sendFn:Ap,now:X,Kc:up,ha:Cl(),Yb:"publicytnetworkstatus-online",Vb:"publicytnetworkstatus-offline",zb:!0,xb:.1,Fb:T("potential_esf_error_limit",10),J:S,lb:!(Uk()&&"www.youtube-nocookie.com"!==bc(document.location.toString()))});this.i=new Bg;S("networkless_immediately_drop_all_requests")&&gp();Ln("LogsDatabaseV2")}
y(zp,No);function Bp(){var a=E("yt.networklessRequestController.instance");a||(a=new zp,F("yt.networklessRequestController.instance",a),S("networkless_logging")&&Rm().then(function(b){a.I=b;Po(a);a.i.resolve();a.zb&&Math.random()<=a.xb&&a.I&&lp(a.I);S("networkless_immediately_drop_sw_health_store")&&Cp(a)}));
return a}
zp.prototype.writeThenSend=function(a,b){b||(b={});Uk()||(this.g=!1);No.prototype.writeThenSend.call(this,a,b)};
zp.prototype.sendThenWrite=function(a,b,c){b||(b={});Uk()||(this.g=!1);No.prototype.sendThenWrite.call(this,a,b,c)};
zp.prototype.sendAndWrite=function(a,b){b||(b={});Uk()||(this.g=!1);No.prototype.sendAndWrite.call(this,a,b)};
zp.prototype.awaitInitialization=function(){return this.i.promise};
function Cp(a){var b;B(function(c){if(!a.I)throw b=Xl("clearSWHealthLogsDb"),b;return c.return(mp(a.I).catch(function(d){a.handleError(d)}))})}
function Ap(a,b,c,d){d=void 0===d?!1:d;b=S("web_fp_via_jspb")?Object.assign({},b):b;S("use_cfr_monitor")&&Dp(a,b);if(S("use_request_time_ms_header"))b.headers&&(b.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(X())));else{var e;if(null==(e=b.postParams)?0:e.requestTimeMs)b.postParams.requestTimeMs=Math.round(X())}if(c&&0===Object.keys(b).length){var f=void 0===f?"":f;var g=void 0===g?!1:g;var h=void 0===h?!1:h;if(a)if(f)tk(a,void 0,"POST",f);else if(R("USE_NET_AJAX_FOR_PING_TRANSPORT",!1)||
h)tk(a,void 0,"GET","",void 0,void 0,g,h);else{b:{try{var k=new $a({url:a});if(k.i&&k.h||k.l){var l=ac(a.match($b)[5]||null);var n=!(!l||!l.endsWith("/aclk")||"1"!==gc(a,"ri"));break b}}catch(q){}n=!1}if(n){b:{try{if(window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,"")){var m=!0;break b}}catch(q){}m=!1}c=m?!0:!1}else c=!1;c||pp(a)}}else b.compress?b.postBody?("string"!==typeof b.postBody&&(b.postBody=JSON.stringify(b.postBody)),Fo(a,b.postBody,b,qk,d)):Fo(a,JSON.stringify(b.postParams),
b,yk,d):qk(a,b)}
function Dp(a,b){var c=b.onError?b.onError:function(){};
b.onError=function(e,f){rp().requestComplete(a,!1);c(e,f)};
var d=b.onSuccess?b.onSuccess:function(){};
b.onSuccess=function(e,f){rp().requestComplete(a,!0);d(e,f)}}
;var Ep=D.ytNetworklessLoggingInitializationOptions||{isNwlInitialized:!1};F("ytNetworklessLoggingInitializationOptions",Ep);function Fp(a){var b=this;this.config_=null;a?this.config_=a:$n()&&(this.config_=ao());Xk(function(){Lo(b)},5E3)}
Fp.prototype.isReady=function(){!this.config_&&$n()&&(this.config_=ao());return!!this.config_};
function Mo(a,b,c,d){function e(t){t=void 0===t?!1:t;var u;if(d.retry&&"www.youtube-nocookie.com"!=h&&(t||S("skip_ls_gel_retry")||"application/json"!==g.headers["Content-Type"]||(u=Jo(b,c,l,k)),u)){var z=g.onSuccess,C=g.onFetchSuccess;g.onSuccess=function(Y,La){Ko(u);z(Y,La)};
c.onFetchSuccess=function(Y,La){Ko(u);C(Y,La)}}try{if(t&&d.retry&&!d.networklessOptions.bypassNetworkless)g.method="POST",d.networklessOptions.writeThenSend?Bp().writeThenSend(p,g):Bp().sendAndWrite(p,g);
else if(d.compress){var P=!d.networklessOptions.writeThenSend;if(g.postBody){var U=g.postBody;"string"!==typeof U&&(U=JSON.stringify(g.postBody));Fo(p,U,g,qk,P)}else Fo(p,JSON.stringify(g.postParams),g,yk,P)}else S("web_all_payloads_via_jspb")?qk(p,g):yk(p,g)}catch(Y){if("InvalidAccessError"==Y.name)u&&(Ko(u),u=0),Kj(Error("An extension is blocking network request."));else throw Y;}u&&Xk(function(){Lo(a)},5E3)}
!R("VISITOR_DATA")&&"visitor_id"!==b&&.01>Math.random()&&Kj(new Tk("Missing VISITOR_DATA when sending innertube request.",b,c,d));if(!a.isReady()){var f=new Tk("innertube xhrclient not ready",b,c,d);Jj(f);throw f;}var g={headers:d.headers||{},method:"POST",postParams:c,postBody:d.postBody,postBodyFormat:d.postBodyFormat||"JSON",onTimeout:function(){d.onTimeout()},
onFetchTimeout:d.onTimeout,onSuccess:function(t,u){if(d.onSuccess)d.onSuccess(u)},
onFetchSuccess:function(t){if(d.onSuccess)d.onSuccess(t)},
onError:function(t,u){if(d.onError)d.onError(u)},
onFetchError:function(t){if(d.onError)d.onError(t)},
timeout:d.timeout,withCredentials:!0,compress:d.compress};g.headers["Content-Type"]||(g.headers["Content-Type"]="application/json");var h="";(f=a.config_.kd)&&(h=f);var k=a.config_.md||!1,l=jo(k,h,d);Object.assign(g.headers,l);(f=g.headers.Authorization)&&!h&&k&&(g.headers["x-origin"]=window.location.origin);var n="/youtubei/"+a.config_.hd+"/"+b,m={alt:"json"},q=a.config_.ld&&f;q=q&&f.startsWith("Bearer");q||(m.key=a.config_.gd);var p=ek(""+h+n,m||{},!0);E("ytNetworklessLoggingInitializationOptions")&&
Ep.isNwlInitialized?Pm().then(function(t){e(t)}):e(!1)}
;function Gp(){var a=E("_lact",window);return null==a?-1:Math.max(Date.now()-a,0)}
;var Hp=D.ytPubsubPubsubInstance||new M,Ip=D.ytPubsubPubsubSubscribedKeys||{},Jp=D.ytPubsubPubsubTopicToKeys||{},Kp=D.ytPubsubPubsubIsSynchronous||{};M.prototype.subscribe=M.prototype.subscribe;M.prototype.unsubscribeByKey=M.prototype.rb;M.prototype.publish=M.prototype.ab;M.prototype.clear=M.prototype.clear;F("ytPubsubPubsubInstance",Hp);F("ytPubsubPubsubTopicToKeys",Jp);F("ytPubsubPubsubIsSynchronous",Kp);F("ytPubsubPubsubSubscribedKeys",Ip);var Lp=Symbol("injectionDeps");function Mp(){this.key=Xn}
function Np(){this.h=new Map;this.g=new Map}
Np.prototype.resolve=function(a){return a instanceof Mp?Op(this,a.key,[],!0):Op(this,a,[])};
function Op(a,b,c,d){d=void 0===d?!1:d;if(-1<c.indexOf(b))throw Error("Deps cycle for: "+b);if(a.g.has(b))return a.g.get(b);if(!a.h.has(b)){if(d)return;throw Error("No provider for: "+b);}d=a.h.get(b);c.push(b);if(void 0!==d.Cd)var e=d.Cd;else if(d.Bd)e=d[Lp]?Pp(a,d[Lp],c):[],e=d.Bd.apply(d,x(e));else if(d.Ad){e=d.Ad;var f=e[Lp]?Pp(a,e[Lp],c):[];e=new (Function.prototype.bind.apply(e,[null].concat(x(f))))}else throw Error("Could not resolve providers for: "+b);c.pop();d.Ae||a.g.set(b,e);return e}
function Pp(a,b,c){return b?b.map(function(d){return d instanceof Mp?Op(a,d.key,c,!0):Op(a,d,c)}):[]}
;var Qp;function Rp(){Qp||(Qp=new Np);return Qp}
;var Sp=window;function Tp(){var a,b;return"h5vcc"in Sp&&(null==(a=Sp.h5vcc.traceEvent)?0:a.traceBegin)&&(null==(b=Sp.h5vcc.traceEvent)?0:b.traceEnd)?1:"performance"in Sp&&Sp.performance.mark&&Sp.performance.measure?2:0}
function Up(a){switch(Tp()){case 1:Sp.h5vcc.traceEvent.traceBegin("YTLR",a);break;case 2:Sp.performance.mark(a+"-start");break;case 0:break;default:we()}}
function Vp(a){switch(Tp()){case 1:Sp.h5vcc.traceEvent.traceEnd("YTLR",a);break;case 2:var b=a+"-start",c=a+"-end";Sp.performance.mark(c);Sp.performance.measure(a,b,c);break;case 0:break;default:we()}}
;var Wp=S("web_enable_lifecycle_monitoring")&&0!==Tp(),Xp=S("web_enable_lifecycle_monitoring");function Yp(a){var b=this;var c=void 0===c?0:c;var d=void 0===d?Cl():d;this.l=c;this.h=d;this.i=new Bg;this.g=a;for(a={Ja:0};a.Ja<this.g.length;a={Eb:void 0,Ja:a.Ja},a.Ja++)a.Eb=this.g[a.Ja],c=function(e){return function(){e.Eb.Ob();b.g[e.Ja].Gb=!0;b.g.every(function(f){return!0===f.Gb})&&b.i.resolve()}}(a),d=this.h.Ha(c,Zp(this,a.Eb)),this.g[a.Ja]=Object.assign({},a.Eb,{Ob:c,
jobId:d})}
function $p(a){var b=Array.from(a.g.keys()).sort(function(d,e){return Zp(a,a.g[e])-Zp(a,a.g[d])});
b=w(b);for(var c=b.next();!c.done;c=b.next())c=a.g[c.value],void 0===c.jobId||c.Gb||(a.h.ba(c.jobId),a.h.Ha(c.Ob,10))}
Yp.prototype.cancel=function(){for(var a=w(this.g),b=a.next();!b.done;b=a.next())b=b.value,void 0===b.jobId||b.Gb||this.h.ba(b.jobId),b.Gb=!0;this.i.resolve()};
function Zp(a,b){var c;return null!=(c=b.priority)?c:a.l}
;function aq(a){this.state=a;this.i=[];this.o=void 0;this.G={};Wp&&Up(this.state)}
aq.prototype.install=function(a){this.i.push(a);return this};
function bq(a){Wp&&Vp(a.state);var b=a.transitions.find(function(d){return Array.isArray(d.from)?d.from.find(function(e){return e===a.state&&"none"===d.Ea}):d.from===a.state&&"none"===d.Ea});
if(b){a.h&&($p(a.h),a.h=void 0);Xp&&console.groupCollapsed&&console.groupEnd&&(console.groupCollapsed("["+a.constructor.name+"] '"+a.state+"' to 'none'"),console.log("with message: ",void 0),console.groupEnd());a.state="none";Wp&&Up(a.state);b=b.action.bind(a);var c=a.i.filter(function(d){return d.none}).map(function(d){return d.none});
b(cq(a,c),void 0)}else throw Error("no transition specified from "+a.state+" to none");}
function cq(a,b){var c=b.filter(function(e){return 10===dq(a,e)}),d=b.filter(function(e){return 10!==dq(a,e)});
return a.G.ze?function(){var e=Ga.apply(0,arguments);return B(function(f){if(1==f.g)return A(f,a.oa.apply(a,[c].concat(x(e))),2);a.u.apply(a,[d].concat(x(e)));f.g=0})}:function(){var e=Ga.apply(0,arguments);
a.ya.apply(a,[c].concat(x(e)));a.u.apply(a,[d].concat(x(e)))}}
aq.prototype.ya=function(a){for(var b=Ga.apply(1,arguments),c=Cl(),d=w(a),e=d.next(),f={};!e.done;f={mb:void 0},e=d.next())f.mb=e.value,c.gb(function(g){return function(){eq(g.mb.name);g.mb.ib.apply(g.mb,x(b));fq(g.mb.name)}}(f))};
aq.prototype.oa=function(a){var b=Ga.apply(1,arguments),c,d,e,f,g;return B(function(h){1==h.g&&(c=Cl(),d=w(a),e=d.next(),f={});if(3!=h.g){if(e.done)return h.B(0);f.Va=e.value;f.tb=void 0;g=function(k){return function(){eq(k.Va.name);var l=k.Va.ib.apply(k.Va,x(b));"function"===typeof(null==l?void 0:l.then)?k.tb=l.then(function(){fq(k.Va.name)}):fq(k.Va.name)}}(f);
c.gb(g);return f.tb?A(h,f.tb,3):h.B(3)}f={Va:void 0,tb:void 0};e=d.next();return h.B(2)})};
aq.prototype.u=function(a){var b=Ga.apply(1,arguments),c=this,d=a.map(function(e){return{Ob:function(){eq(e.name);e.ib.apply(e,x(b));fq(e.name)},
priority:dq(c,e)}});
d.length&&(this.h=new Yp(d))};
function dq(a,b){var c,d;return null!=(d=null!=(c=a.o)?c:b.priority)?d:0}
function eq(a){Wp&&a&&Up(a)}
function fq(a){Wp&&a&&Vp(a)}
da.Object.defineProperties(aq.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return this.state}}});function gq(a){aq.call(this,void 0===a?"none":a);this.g=null;this.o=10;this.transitions=[{from:"none",Ea:"application_navigating",action:this.K},{from:"application_navigating",Ea:"none",action:this.O},{from:"application_navigating",Ea:"application_navigating",action:function(){}},
{from:"none",Ea:"none",action:function(){}}]}
var hq;y(gq,aq);gq.prototype.K=function(a,b){var c=this;this.g=Xk(function(){"application_navigating"===c.l&&bq(c)},5E3);
a(null==b?void 0:b.event)};
gq.prototype.O=function(a,b){this.g&&(Df.ba(this.g),this.g=null);a(null==b?void 0:b.event)};
function iq(){hq||(hq=new gq);return hq}
;var jq=[];F("yt.logging.transport.getScrapedGelPayloads",function(){return jq});function kq(){this.store={};this.g={}}
kq.prototype.storePayload=function(a,b){a=lq(a);this.store[a]?this.store[a].push(b):(this.g={},this.store[a]=[b]);return a};
kq.prototype.smartExtractMatchingEntries=function(a){if(!a.keys.length)return[];for(var b=mq(this,a.keys.splice(0,1)[0]),c=[],d=0;d<b.length;d++)this.store[b[d]]&&a.sizeLimit&&(this.store[b[d]].length<=a.sizeLimit?(c.push.apply(c,x(this.store[b[d]])),delete this.store[b[d]]):c.push.apply(c,x(this.store[b[d]].splice(0,a.sizeLimit))));(null==a?0:a.sizeLimit)&&c.length<(null==a?void 0:a.sizeLimit)&&(a.sizeLimit-=c.length,c.push.apply(c,x(this.smartExtractMatchingEntries(a))));return c};
kq.prototype.extractMatchingEntries=function(a){a=mq(this,a);for(var b=[],c=0;c<a.length;c++)this.store[a[c]]&&(b.push.apply(b,x(this.store[a[c]])),delete this.store[a[c]]);return b};
kq.prototype.getSequenceCount=function(a){a=mq(this,a);for(var b=0,c=0;c<a.length;c++){var d=void 0;b+=(null==(d=this.store[a[c]])?void 0:d.length)||0}return b};
function mq(a,b){var c=lq(b);if(a.g[c])return a.g[c];var d=Object.keys(a.store)||[];if(1>=d.length&&lq(b)===d[0])return d;for(var e=[],f=0;f<d.length;f++){var g=d[f].split("/");if(nq(b.auth,g[0])){var h=b.isJspb;nq(void 0===h?"undefined":h?"true":"false",g[1])&&nq(b.cttAuthInfo,g[2])&&(h=b.tier,h=void 0===h?"undefined":JSON.stringify(h),nq(h,g[3])&&e.push(d[f]))}}return a.g[c]=e}
function nq(a,b){return void 0===a||"undefined"===a?!0:a===b}
kq.prototype.getSequenceCount=kq.prototype.getSequenceCount;kq.prototype.extractMatchingEntries=kq.prototype.extractMatchingEntries;kq.prototype.smartExtractMatchingEntries=kq.prototype.smartExtractMatchingEntries;kq.prototype.storePayload=kq.prototype.storePayload;function lq(a){return[void 0===a.auth?"undefined":a.auth,void 0===a.isJspb?"undefined":a.isJspb,void 0===a.cttAuthInfo?"undefined":a.cttAuthInfo,void 0===a.tier?"undefined":a.tier].join("/")}
;var oq=T("initial_gel_batch_timeout",2E3),pq=T("gel_queue_timeout_max_ms",6E4),qq=Math.pow(2,16)-1,rq=T("gel_min_batch_size",5),sq=void 0;function tq(){this.l=this.g=this.h=0;this.i=!1}
var uq=new tq,vq=new tq,wq=new tq,xq=new tq,yq,zq=!0,Aq=1,Bq=new Map,Cq=D.ytLoggingTransportTokensToCttTargetIds_||{};F("ytLoggingTransportTokensToCttTargetIds_",Cq);var Dq=D.ytLoggingTransportTokensToJspbCttTargetIds_||{};F("ytLoggingTransportTokensToJspbCttTargetIds_",Dq);var Eq={};function Fq(){var a=E("yt.logging.ims");a||(a=new kq,F("yt.logging.ims",a));return a}
function Gq(a,b){if("log_event"===a.endpoint){Hq(a);var c=Iq(a),d=Jq(a.payload)||"",e=Kq(d),f=200;if(e){if(!1===e.enabled&&!S("web_payload_policy_disabled_killswitch"))return;f=Lq(e.tier);if(400===f){Mq(a,b);return}}Eq[c]=!0;e={cttAuthInfo:c,isJspb:!1,tier:f};Fq().storePayload(e,a.payload);Nq(b,c,!1,e,Oq(d))}}
function Pq(a,b,c){if("log_event"===b.endpoint){Hq(void 0,b);var d=Iq(b,!0),e=Kq(a),f=200;if(e){if(!1===e.enabled&&!S("web_payload_policy_disabled_killswitch"))return;f=Lq(e.tier);if(400===f){Qq(a,b,c);return}}Eq[d]=!0;e={cttAuthInfo:d,isJspb:!0,tier:f};Fq().storePayload(e,b.payload.toJSON());Nq(c,d,!0,e,Oq(a))}}
function Nq(a,b,c,d,e){function f(){Rq({writeThenSend:!0},S("flush_only_full_queue")?b:void 0,c,d.tier)}
c=void 0===c?!1:c;e=void 0===e?!1:e;a&&(sq=new a);a=T("tvhtml5_logging_max_batch_ads_fork")||T("web_logging_max_batch")||100;var g=X(),h=Sq(c,d.tier),k=h.l;e&&(h.i=!0);e=0;d&&(e=Fq().getSequenceCount(d));1E3<=e?f():e>=a?yq||(yq=Tq(function(){f();yq=void 0},0)):10<=g-k&&(Uq(c,d.tier),h.l=g)}
function Mq(a,b){if("log_event"===a.endpoint){Hq(a);var c=Iq(a),d=new Map;d.set(c,[a.payload]);var e=Jq(a.payload)||"";b&&(sq=new b);return new Cg(function(f,g){sq&&sq.isReady()?Vq(d,sq,f,g,{bypassNetworkless:!0},!0,Oq(e)):f()})}}
function Qq(a,b,c){if("log_event"===b.endpoint){Hq(void 0,b);var d=Iq(b,!0),e=new Map;e.set(d,[b.payload.toJSON()]);c&&(sq=new c);return new Cg(function(f){sq&&sq.isReady()?Wq(e,sq,f,{bypassNetworkless:!0},!0,Oq(a)):f()})}}
function Iq(a,b){var c="";if(a.dangerousLogToVisitorSession)c="visitorOnlyApprovedKey";else if(a.cttAuthInfo){if(void 0===b?0:b){b=a.cttAuthInfo.token;c=a.cttAuthInfo;var d=new xj;c.videoId?Vd(d,1,de,Bd(c.videoId)):c.playlistId&&Vd(d,2,de,Bd(c.playlistId));Dq[b]=d}else b=a.cttAuthInfo,c={},b.videoId?c.videoId=b.videoId:b.playlistId&&(c.playlistId=b.playlistId),Cq[a.cttAuthInfo.token]=c;c=a.cttAuthInfo.token}return c}
function Rq(a,b,c,d){a=void 0===a?{}:a;c=void 0===c?!1:c;new Cg(function(e,f){var g=Sq(c,d),h=g.i;g.i=!1;Xq(g.h);Xq(g.g);g.g=0;sq&&sq.isReady()?void 0===d&&S("enable_web_tiered_gel")?Yq(e,f,a,b,c,300,h):Yq(e,f,a,b,c,d,h):(Uq(c,d),e())})}
function Yq(a,b,c,d,e,f,g){var h=sq;c=void 0===c?{}:c;e=void 0===e?!1:e;f=void 0===f?200:f;g=void 0===g?!1:g;var k=new Map,l=new Map,n={isJspb:e,cttAuthInfo:d,tier:f},m={isJspb:e,cttAuthInfo:d};if(void 0!==d)e?(b=S("enable_web_tiered_gel")?Fq().smartExtractMatchingEntries({keys:[n,m],sizeLimit:1E3}):Fq().extractMatchingEntries(m),k.set(d,b),Wq(k,h,a,c,!1,g)):(k=S("enable_web_tiered_gel")?Fq().smartExtractMatchingEntries({keys:[n,m],sizeLimit:1E3}):Fq().extractMatchingEntries(m),l.set(d,k),Vq(l,h,
a,b,c,!1,g));else if(e){b=w(Object.keys(Eq));for(d=b.next();!d.done;d=b.next())l=d.value,f=S("enable_web_tiered_gel")?Fq().smartExtractMatchingEntries({keys:[n,m],sizeLimit:1E3}):Fq().extractMatchingEntries({isJspb:!0,cttAuthInfo:l}),0<f.length&&k.set(l,f),(S("web_fp_via_jspb_and_json")&&c.writeThenSend||!S("web_fp_via_jspb_and_json"))&&delete Eq[l];Wq(k,h,a,c,!1,g)}else{k=w(Object.keys(Eq));for(d=k.next();!d.done;d=k.next())n=d.value,m=S("enable_web_tiered_gel")?Fq().smartExtractMatchingEntries({keys:[{isJspb:!1,
cttAuthInfo:n,tier:f},{isJspb:!1,cttAuthInfo:n}],sizeLimit:1E3}):Fq().extractMatchingEntries({isJspb:!1,cttAuthInfo:n}),0<m.length&&l.set(n,m),(S("web_fp_via_jspb_and_json")&&c.writeThenSend||!S("web_fp_via_jspb_and_json"))&&delete Eq[n];Vq(l,h,a,b,c,!1,g)}}
function Uq(a,b){function c(){Rq({writeThenSend:!0},void 0,a,b)}
a=void 0===a?!1:a;b=void 0===b?200:b;var d=Sq(a,b),e=d===xq||d===wq?5E3:pq;S("web_gel_timeout_cap")&&!d.g&&(e=Tq(function(){c()},e),d.g=e);
Xq(d.h);e=R("LOGGING_BATCH_TIMEOUT",T("web_gel_debounce_ms",1E4));S("shorten_initial_gel_batch_timeout")&&zq&&(e=oq);e=Tq(function(){0<T("gel_min_batch_size")?Fq().getSequenceCount({cttAuthInfo:void 0,isJspb:a,tier:b})>=rq&&c():c()},e);
d.h=e}
function Vq(a,b,c,d,e,f,g){e=void 0===e?{}:e;var h=Math.round(X()),k=a.size,l=Zq(g);a=w(a);var n=a.next();for(g={};!n.done;g={Ub:void 0,batchRequest:void 0,dangerousLogToVisitorSession:void 0,Xb:void 0,Wb:void 0},n=a.next()){var m=w(n.value);n=m.next().value;m=m.next().value;g.batchRequest=pb({context:bo(b.config_||ao())});if(!Na(m)&&!S("throw_err_when_logevent_malformed_killswitch")){d();break}g.batchRequest.events=m;(m=Cq[n])&&$q(g.batchRequest,n,m);delete Cq[n];g.dangerousLogToVisitorSession="visitorOnlyApprovedKey"===
n;ar(g.batchRequest,h,g.dangerousLogToVisitorSession);br(e);g.Xb=function(q){S("start_client_gcf")&&Df.qa(function(){return B(function(p){return A(p,cr(q),0)})});
k--;k||c()};
g.Ub=0;g.Wb=function(q){return function(){q.Ub++;if(e.bypassNetworkless&&1===q.Ub)try{Mo(b,l,q.batchRequest,dr({writeThenSend:!0},q.dangerousLogToVisitorSession,q.Xb,q.Wb,f)),zq=!1}catch(p){Jj(p),d()}k--;k||c()}}(g);
try{Mo(b,l,g.batchRequest,dr(e,g.dangerousLogToVisitorSession,g.Xb,g.Wb,f)),zq=!1}catch(q){Jj(q),d()}}}
function Wq(a,b,c,d,e,f){d=void 0===d?{}:d;var g=Math.round(X()),h={value:a.size},k=new Map([].concat(x(a)));k=w(k);for(var l=k.next();!l.done;l=k.next()){var n=w(l.value).next().value,m=a.get(n);l=new yj;var q=b.config_||ao(),p=new gj,t=new $i;K(t,1,q.wc);K(t,2,q.vc);fe(t,16,q.jd);K(t,17,q.uc);if(q.Mb){var u=q.Mb,z=new Xi;u.coldConfigData&&K(z,1,u.coldConfigData);u.appInstallData&&K(z,6,u.appInstallData);u.coldHashData&&K(z,3,u.coldHashData);u.hotHashData&&z.g(u.hotHashData);J(t,Xi,62,z)}if((u=D.devicePixelRatio)&&
1!=u){if(null!=u&&"number"!==typeof u)throw Error("Value of float/double field must be a number, found "+typeof u+": "+u);Sd(t,65,u)}u=R("EXPERIMENTS_TOKEN","");""!==u&&K(t,54,u);u=Mj();if(0<u.length){z=new cj;for(var C=0;C<u.length;C++){var P=new aj;K(P,1,u[C].key);Vd(P,2,bj,Bd(u[C].value));ae(z,15,aj,P)}J(p,cj,5,z)}co(q,t);eo(p);fo(t);go(q,t);ho(t);S("start_client_gcf")&&io(t);R("DELEGATED_SESSION_ID")&&!S("pageid_as_header_web")&&(q=new fj,K(q,3,R("DELEGATED_SESSION_ID")));!S("fill_delegate_context_in_gel_killswitch")&&
(u=R("INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT"))&&(z=Yd(p,fj,3)||new fj,q=p,u=K(z,18,u),J(q,fj,3,u));q=t;u=w(Object.entries(dk(R("DEVICE",""))));for(z=u.next();!z.done;z=u.next())C=w(z.value),z=C.next().value,C=C.next().value,"cbrand"===z?K(q,12,C):"cmodel"===z?K(q,13,C):"cbr"===z?K(q,87,C):"cbrver"===z?K(q,88,C):"cos"===z?K(q,18,C):"cosver"===z?K(q,19,C):"cplatform"===z&&fe(q,42,Nk(C));J(p,$i,1,t);J(l,gj,1,p);if(t=Dq[n])a:{if(ce(t,1))p=1;else if(t.getPlaylistId())p=2;else break a;J(l,xj,
4,t);t=Yd(l,gj,1)||new gj;q=Yd(t,fj,3)||new fj;u=new ej;K(u,2,n);fe(u,1,p);ae(q,12,ej,u);J(t,fj,3,q)}delete Dq[n];n="visitorOnlyApprovedKey"===n;er()||Sd(l,2,Ad(g));!n&&(p=R("EVENT_ID"))&&(t=fr(),q=new wj,K(q,1,p),Sd(q,2,Ad(t)),J(l,wj,5,q));br(d);if(S("jspb_serialize_with_worker")&&(p=ro())&&d.writeThenSend){Bq.set(Aq,{client:b,resolve:c,networklessOptions:d,isIsolated:e,useVSSEndpoint:f,dangerousLogToVisitorSession:n,requestsOutstanding:h});p.postMessage({op:"gelBatchToSerialize",batchRequest:l.toJSON(),
clientEvents:m,key:Aq});Aq++;break}if(m){p=[];for(t=0;t<m.length;t++)try{p.push(new uj(m[t]))}catch(U){Jj(new Tk("Transport failed to deserialize "+String(m[t])))}m=p}else m=[];m=w(m);for(p=m.next();!p.done;p=m.next())ae(l,3,uj,p.value);m={startTime:X(),ticks:{},infos:{}};l=he(l);m.ticks.geljspc=X();S("log_jspb_serialize_latency")&&no("gel_jspb_serialize",m,{sampleRate:.1});gr(l,b,c,d,e,f,n,h)}}
function gr(a,b,c,d,e,f,g,h){d=void 0===d?{}:d;h=void 0===h?{value:0}:h;f=Zq(f);d=dr(d,g,function(k){S("start_client_gcf")&&Df.qa(function(){return B(function(l){return A(l,cr(k),0)})});
h.value--;h.value||c()},function(){h.value--;
h.value||c()},e);
d.headers["Content-Type"]="application/json+protobuf";d.postBodyFormat="JSPB";d.postBody=a;Mo(b,f,"",d);zq=!1}
function br(a){S("always_send_and_write")&&(a.writeThenSend=!1)}
function dr(a,b,c,d,e){a={retry:!0,onSuccess:c,onError:d,networklessOptions:a,dangerousLogToVisitorSession:b,qe:!!e,headers:{},postBodyFormat:"",postBody:"",compress:S("compress_gel")||S("compress_gel_lr")};er()&&(a.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(X())));return a}
function ar(a,b,c){er()||(a.requestTimeMs=String(b));S("unsplit_gel_payloads_in_logs")&&(a.unsplitGelPayloadsInLogs=!0);!c&&(b=R("EVENT_ID"))&&(c=fr(),a.serializedClientEventId={serializedEventId:b,clientCounter:String(c)})}
function fr(){var a=R("BATCH_CLIENT_COUNTER")||0;a||(a=Math.floor(Math.random()*qq/2));a++;a>qq&&(a=1);Dj("BATCH_CLIENT_COUNTER",a);return a}
function $q(a,b,c){if(c.videoId)var d="VIDEO";else if(c.playlistId)d="PLAYLIST";else return;a.credentialTransferTokenTargetId=c;a.context=a.context||{};a.context.user=a.context.user||{};a.context.user.credentialTransferTokens=[{token:b,scope:d}]}
function Hq(a,b){if(!E("yt.logging.transport.enableScrapingForTest")){var c=Lj("il_payload_scraping");if("enable_il_payload_scraping"===(void 0!==c?String(c):""))jq=[],F("yt.logging.transport.enableScrapingForTest",!0),F("yt.logging.transport.scrapedPayloadsForTesting",jq),F("yt.logging.transport.payloadToScrape","visualElementShown visualElementHidden visualElementAttached screenCreated visualElementGestured visualElementStateChanged".split(" ")),F("yt.logging.transport.getScrapedPayloadFromClientEventsFunction"),
F("yt.logging.transport.scrapeClientEvent",!0);else return}c=E("yt.logging.transport.scrapedPayloadsForTesting");var d=E("yt.logging.transport.payloadToScrape");b&&(b=E("yt.logging.transport.getScrapedPayloadFromClientEventsFunction").bind(b.payload)())&&c.push(b);b=E("yt.logging.transport.scrapeClientEvent");if(d&&1<=d.length)for(var e=0;e<d.length;e++)if(a&&a.payload[d[e]])if(b)c.push(a.payload);else{var f=void 0;c.push((null==(f=a)?void 0:f.payload)[d[e]])}F("yt.logging.transport.scrapedPayloadsForTesting",
c)}
function er(){return S("use_request_time_ms_header")||S("lr_use_request_time_ms_header")}
function Tq(a,b){return!1===S("embeds_transport_use_scheduler")?Vj(a,b):S("logging_avoid_blocking_during_navigation")||S("lr_logging_avoid_blocking_during_navigation")?Xk(function(){if("none"===iq().l)a();else{var c={};iq().install((c.none={ib:a},c))}},b):Xk(a,b)}
function Xq(a){S("transport_use_scheduler")?Df.ba(a):window.clearTimeout(a)}
function cr(a){var b,c,d,e,f,g,h,k,l,n;return B(function(m){if(1==m.g){d=null==(b=a)?void 0:null==(c=b.responseContext)?void 0:c.globalConfigGroup;var q=d?d[Wi.name]:void 0;e=q;g=null==(f=d)?void 0:f.hotHashData;q=d?d[Vi.name]:void 0;h=q;l=null==(k=d)?void 0:k.coldHashData;return(n=Rp().resolve(new Mp))?g?e?A(m,Yn(n,g,e),2):A(m,Yn(n,g),2):m.B(2):m.return()}return l?h?A(m,Zn(n,l,h),0):A(m,Zn(n,l),0):m.B(0)})}
function Sq(a,b){b=void 0===b?200:b;return a?300===b?xq:vq:300===b?wq:uq}
function Kq(a){if(S("enable_web_tiered_gel")){a=hp[a||""];var b,c;if(null==Rp().resolve(new Mp))var d=void 0;else{var e=null!=(d=E("yt.gcf.config.hotConfigGroup"))?d:R("RAW_HOT_CONFIG_GROUP");d=null==e?void 0:null==(b=e.loggingHotConfig)?void 0:null==(c=b.eventLoggingConfig)?void 0:c.payloadPolicies}if(b=d)for(c=0;c<b.length;c++)if(b[c].payloadNumber===a)return b[c]}}
function Jq(a){a=Object.keys(a);a=w(a);for(var b=a.next();!b.done;b=a.next())if(b=b.value,hp[b])return b}
function Lq(a){switch(a){case "DELAYED_EVENT_TIER_UNSPECIFIED":return 0;case "DELAYED_EVENT_TIER_DEFAULT":return 100;case "DELAYED_EVENT_TIER_DISPATCH_TO_EMPTY":return 200;case "DELAYED_EVENT_TIER_FAST":return 300;case "DELAYED_EVENT_TIER_IMMEDIATE":return 400;default:return 200}}
function Oq(a){return"gelDebuggingEvent"===a}
function Zq(a){return(void 0===a?0:a)&&S("vss_through_gel_video_stats")?"video_stats":"log_event"}
;var hr=D.ytLoggingGelSequenceIdObj_||{};F("ytLoggingGelSequenceIdObj_",hr);
function ir(a,b,c,d){d=void 0===d?{}:d;var e={},f=Math.round(d.timestamp||X());e.eventTimeMs=f<Number.MAX_SAFE_INTEGER?f:0;e[a]=b;a=Gp();e.context={lastActivityMs:String(d.timestamp||!isFinite(a)?-1:a)};d.sequenceGroup&&!S("web_gel_sequence_info_killswitch")&&(a=e.context,b=d.sequenceGroup,b={index:jr(b),groupKey:b},a.sequence=b,d.endOfSequence&&delete hr[d.sequenceGroup]);(d.sendIsolatedPayload?Mq:Gq)({endpoint:"log_event",payload:e,cttAuthInfo:d.cttAuthInfo,dangerousLogToVisitorSession:d.dangerousLogToVisitorSession},
c)}
function kr(a){Rq(void 0,void 0,void 0===a?!1:a)}
function jr(a){hr[a]=a in hr?hr[a]+1:0;return hr[a]}
;var lr=[];function mr(a,b,c){c=void 0===c?{}:c;var d=Fp;R("ytLoggingEventsDefaultDisabled",!1)&&Fp===Fp&&(d=null);S("web_all_payloads_via_jspb")?(c.timestamp||(c.lact=Gp(),c.timestamp=X()),lr.push({Ac:a,payload:b,options:c})):ir(a,b,d,c)}
;var nr=D.ytLoggingGelSequenceIdObj_||{};F("ytLoggingGelSequenceIdObj_",nr);function or(a,b){var c=void 0;c=void 0===c?{}:c;var d=!1;R("ytLoggingEventsDefaultDisabled",!1)&&(d=!0);d=d?null:Fp;c=void 0===c?{}:c;var e=Math.round(c.timestamp||X());Sd(b,1,Ad(e<Number.MAX_SAFE_INTEGER?e:0));e=new tj;if(c.lact)Sd(e,1,Ad(isFinite(c.lact)?c.lact:-1));else if(c.timestamp)Sd(e,1,Ad(-1));else{var f=Gp();Sd(e,1,Ad(isFinite(f)?f:-1))}if(c.sequenceGroup&&!S("web_gel_sequence_info_killswitch")){f=c.sequenceGroup;var g=jr(f),h=new sj;Sd(h,2,Ad(g));K(h,1,f);J(e,sj,3,h);c.endOfSequence&&delete nr[c.sequenceGroup]}J(b,
tj,33,e);(c.sendIsolatedPayload?Qq:Pq)(a,{endpoint:"log_event",payload:b,cttAuthInfo:c.cttAuthInfo,dangerousLogToVisitorSession:c.dangerousLogToVisitorSession},d)}
;var pr=new Set,qr=0,rr=0,sr=0,tr=[],ur=["PhantomJS","Googlebot","TO STOP THIS SECURITY SCAN go/scan"];function vr(a){try{pr.add(a.message)}catch(b){}qr++}
function wr(){for(var a=w(ur),b=a.next();!b.done;b=a.next()){var c=Mb();if(c&&0<=c.toLowerCase().indexOf(b.value.toLowerCase()))return!0}return!1}
function xr(a,b){var c=b.params||{};a={urlParams:{a:"logerror",t:"jserror",type:b.name,msg:b.message.substr(0,250),line:b.lineNumber,level:a,"client.name":c.name},postParams:{url:R("PAGE_NAME",window.location.href),file:b.fileName},method:"POST"};c.version&&(a["client.version"]=c.version);if(a.postParams){b.stack&&(a.postParams.stack=b.stack);b=w(Object.keys(c));for(var d=b.next();!d.done;d=b.next())d=d.value,a.postParams["client."+d]=c[d];if(c=R("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS"))for(b=w(Object.keys(c)),
d=b.next();!d.done;d=b.next())d=d.value,a.postParams[d]=c[d];c=R("SERVER_NAME");b=R("SERVER_VERSION");c&&b&&(a.postParams["server.name"]=c,a.postParams["server.version"]=b)}qk(R("ECATCHER_REPORT_HOST","")+"/error_204",a)}
;function yr(){var a;return B(function(b){return(a=Wf())?b.return(a.then(function(c){c=he(c);for(var d=[],e=0,f=0;f<c.length;f++){var g=c.charCodeAt(f);255<g&&(d[e++]=g&255,g>>=8);d[e++]=g}return Hc(d,3)})):b.return(Promise.resolve(null))})}
;var zr={};function Ar(a){return zr[a]||(zr[a]=String(a).replace(/\-([a-z])/g,function(b,c){return c.toUpperCase()}))}
;var Br={},Cr=[],Tg=new M,Dr={};function Er(){for(var a=w(Cr),b=a.next();!b.done;b=a.next())b=b.value,b()}
function Fr(a,b){var c;"yt:"===a.tagName.toLowerCase().substr(0,3)?c=a.getAttribute(b):c=a?a.dataset?a.dataset[Ar(b)]:a.getAttribute("data-"+b):null;return c}
function Gr(a){Tg.ab.apply(Tg,arguments)}
;function Hr(a){this.g=a||{};a=[this.g,window.YTConfig||{}];for(var b=0;b<a.length;b++)a[b].host&&(a[b].host=a[b].host.toString().replace("http://","https://"))}
function Ir(a,b){a=[a.g,window.YTConfig||{}];for(var c=0;c<a.length;c++){var d=a[c][b];if(void 0!==d)return d}return null}
function Jr(a,b,c){Kr||(Kr={},Lr=new Set,Uj(window,"message",function(d){a:if(Lr.has(d.origin)){try{var e=JSON.parse(d.data)}catch(g){break a}var f=Kr[e.id];f&&d.origin===f.Sc&&(d=f.Dd,d.G=!0,d.G&&(db(d.u,d.sendMessage,d),d.u.length=0),d.fc(e))}}));
a=String(Ir(a,"host"));Kr[c]={Dd:b,Sc:a};Lr.add(a)}
var Kr=null,Lr=null;var Mr=window;
function Nr(a,b,c){this.o=this.g=this.h=null;this.i=0;this.G=!1;this.u=[];this.l=null;this.O={};if(!a)throw Error("YouTube player element ID required.");this.id=Qa(this);this.K=c;c=document;if(a="string"===typeof a?c.getElementById(a):a)if(c="iframe"===a.tagName.toLowerCase(),b.host||(b.host=c?cc(a.src):"https://www.youtube.com"),this.h=new Hr(b),c||(b=Or(this,a),this.o=a,(c=a.parentNode)&&c.replaceChild(b,a),a=b),this.g=a,this.g.id||(this.g.id="widget"+Qa(this.g)),Br[this.g.id]=this,window.postMessage){this.l=
new M;Pr(this);b=Ir(this.h,"events");for(var d in b)b.hasOwnProperty(d)&&this.addEventListener(d,b[d]);for(var e in Dr)Dr.hasOwnProperty(e)&&Qr(this,e)}}
r=Nr.prototype;r.setSize=function(a,b){this.g.width=a.toString();this.g.height=b.toString();return this};
r.getIframe=function(){return this.g};
r.fc=function(a){Rr(this,a.event,a)};
r.addEventListener=function(a,b){var c=b;"string"===typeof b&&(c=function(){window[b].apply(window,arguments)});
if(!c)return this;this.l.subscribe(a,c);Sr(this,a);return this};
function Qr(a,b){b=b.split(".");if(2===b.length){var c=b[1];a.K===b[0]&&Sr(a,c)}}
r.destroy=function(){this.g&&this.g.id&&(Br[this.g.id]=null);Se(this.l);if(this.o){var a=this.o,b=this.g,c=b.parentNode;c&&c.replaceChild(a,b)}else(a=this.g)&&a.parentNode&&a.parentNode.removeChild(a);Kr&&(Kr[this.id]=null);this.h=null;a=this.g;for(var d in nb)nb[d][0]==a&&Sj(d);this.o=this.g=null};
r.kc=function(){return{}};
function Tr(a,b,c){c=c||[];c=Array.prototype.slice.call(c);b={event:"command",func:b,args:c};a.G?a.sendMessage(b):a.u.push(b)}
function Rr(a,b,c){a.l.Aa||(c={target:a,data:c},a.l.ab(b,c),Gr(a.K+"."+b,c))}
function Or(a,b){var c=document.createElement("iframe");b=b.attributes;for(var d=0,e=b.length;d<e;d++){var f=b[d].value;null!=f&&""!==f&&"null"!==f&&c.setAttribute(b[d].name,f)}c.setAttribute("frameBorder","0");c.setAttribute("allowfullscreen","");c.setAttribute("allow","accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share");c.setAttribute("title","YouTube "+Ir(a.h,"title"));(b=Ir(a.h,"width"))&&c.setAttribute("width",b.toString());(b=Ir(a.h,"height"))&&
c.setAttribute("height",b.toString());Mr.yt_embedsEnableIframeWithLazyLoad&&c.setAttribute("loading","lazy");var g=a.kc();g.enablejsapi=window.postMessage?1:0;window.location.host&&(g.origin=window.location.protocol+"//"+window.location.host);g.widgetid=a.id;window.location.href&&db(["debugjs","debugcss"],function(k){var l=gc(window.location.href,k);null!==l&&(g[k]=l)});
var h=""+Ir(a.h,"host")+("/embed/"+Ir(a.h,"videoId"))+"?"+ec(g);Mr.yt_embedsEnableUaChProbe?yr().then(function(k){var l=new URL(h),n=Number(l.searchParams.get("reloads"));isNaN(n)&&(n=0);l.searchParams.set("reloads",(n+1).toString());k&&l.searchParams.set("uach",k);l.searchParams.set("uats",Math.floor(window.performance.timeOrigin).toString());k=te(l.href).toString();ze(c,ue(k));c.sandbox.add("allow-presentation","allow-top-navigation");return k}):Mr.yt_embedsEnableIframeSrcWithIntent?(ze(c,ue(h)),
c.sandbox.add("allow-presentation","allow-top-navigation")):c.src=h;
return c}
r.Bc=function(){this.g&&this.g.contentWindow?this.sendMessage({event:"listening"}):window.clearInterval(this.i)};
function Pr(a){Jr(a.h,a,a.id);a.i=Wj(a.Bc.bind(a));Uj(a.g,"load",function(){window.clearInterval(a.i);a.i=Wj(a.Bc.bind(a))})}
function Sr(a,b){a.O[b]||(a.O[b]=!0,Tr(a,"addEventListener",[b]))}
r.sendMessage=function(a){a.id=this.id;a.channel="widget";var b=JSON.stringify(a),c=[cc(this.g.src||"").replace("http:","https:")];if(this.g.contentWindow)for(var d=0;d<c.length;d++)try{this.g.contentWindow.postMessage(b,c[d])}catch(zc){if(zc.name&&"SyntaxError"===zc.name){if(!(zc.message&&0<zc.message.indexOf("target origin ''"))){var e=void 0,f=zc;e=void 0===e?{}:e;e.name=R("INNERTUBE_CONTEXT_CLIENT_NAME",1);e.version=R("INNERTUBE_CONTEXT_CLIENT_VERSION");var g="WARNING",h=!1;g=void 0===g?"ERROR":
g;h=void 0===h?!1:h;if(f){f.hasOwnProperty("level")&&f.level&&(g=f.level);if(S("console_log_js_exceptions")){var k=f,l=[];l.push("Name: "+k.name);l.push("Message: "+k.message);k.hasOwnProperty("params")&&l.push("Error Params: "+JSON.stringify(k.params));k.hasOwnProperty("args")&&l.push("Error args: "+JSON.stringify(k.args));l.push("File name: "+k.fileName);l.push("Stacktrace: "+k.stack);window.console.log(l.join("\n"),k)}if(!(5<=qr)){var n=void 0,m=void 0,q=f,p=e,t=me(q),u=t.message||"Unknown Error",
z=t.name||"UnknownError",C=t.stack||q.h||"Not available";if(C.startsWith(z+": "+u)){var P=C.split("\n");P.shift();C=P.join("\n")}var U=t.lineNumber||"Not available",Y=t.fileName||"Not available",La=C,Ca=0;if(q.hasOwnProperty("args")&&q.args&&q.args.length)for(var Ma=0;Ma<q.args.length&&!(Ca=Ik(q.args[Ma],"params."+Ma,p,Ca),500<=Ca);Ma++);else if(q.hasOwnProperty("params")&&q.params){var ja=q.params;if("object"===typeof q.params)for(m in ja){if(ja[m]){var ra="params."+m,sa=Kk(ja[m]);p[ra]=sa;Ca+=ra.length+
sa.length;if(500<Ca)break}}else p.params=Kk(ja)}if(tr.length)for(var ia=0;ia<tr.length&&!(Ca=Ik(tr[ia],"params.context."+ia,p,Ca),500<=Ca);ia++);navigator.vendor&&!p.hasOwnProperty("vendor")&&(p["device.vendor"]=navigator.vendor);var W={message:u,name:z,lineNumber:U,fileName:Y,stack:La,params:p,sampleWeight:1},Xm=Number(q.columnNumber);isNaN(Xm)||(W.lineNumber=W.lineNumber+":"+Xm);if("IGNORED"===q.level)n=0;else a:{for(var Ym=Ek(),Zm=w(Ym.Ca),Rh=Zm.next();!Rh.done;Rh=Zm.next()){var $m=Rh.value;if(W.message&&
W.message.match($m.ue)){n=$m.weight;break a}}for(var an=w(Ym.za),Sh=an.next();!Sh.done;Sh=an.next()){var bn=Sh.value;if(bn.ib(W)){n=bn.weight;break a}}n=1}W.sampleWeight=n;for(var cn=w(zk),Th=cn.next();!Th.done;Th=cn.next()){var Uh=Th.value;if(Uh.Db[W.name])for(var dn=w(Uh.Db[W.name]),Vh=dn.next();!Vh.done;Vh=dn.next()){var en=Vh.value,Te=W.message.match(en.regexp);if(Te){W.params["params.error.original"]=Te[0];for(var Wh=en.groups,fn={},Ac=0;Ac<Wh.length;Ac++)fn[Wh[Ac]]=Te[Ac+1],W.params["params.error."+
Wh[Ac]]=Te[Ac+1];W.message=Uh.Sb(fn);break}}}W.params||(W.params={});var gn=Ek();W.params["params.errorServiceSignature"]="msg="+gn.Ca.length+"&cb="+gn.za.length;W.params["params.serviceWorker"]="false";D.document&&D.document.querySelectorAll&&(W.params["params.fscripts"]=String(document.querySelectorAll("script:not([nonce])").length));wb("sample").constructor!==ub&&(W.params["params.fconst"]="true");var sd=W;window.yterr&&"function"===typeof window.yterr&&window.yterr(sd);if(0!==sd.sampleWeight&&
!pr.has(sd.message))if(h&&S("web_enable_error_204")){var hn=sd;xr(void 0===g?"ERROR":g,hn);vr(hn)}else{var Xh=void 0,Yh=void 0,jn=void 0,kn=void 0,Zh=void 0,O=sd,Tb=g;Tb=void 0===Tb?"ERROR":Tb;if("ERROR"===Tb){Fk.ab("handleError",O);if(S("record_app_crashed_web")&&0===sr&&1===O.sampleWeight)if(sr++,S("errors_via_jspb")){var Zr=new rj;Zh=fe(Zr,1,1);if(!S("report_client_error_with_app_crash_ks")){var $r=new qj,as=new pj,bs=new oj,cs=new nj;var ds=K(cs,1,O.message);var es=J(bs,nj,3,ds);kn=J(as,oj,5,
es);jn=J($r,pj,9,kn);J(Zh,qj,4,jn)}var ln=S("jspb_sparse_encoded_pivot")?new uj([{}]):new uj;Zd(ln,rj,20,vj,Zh);or("appCrashed",ln)}else{var mn={appCrashType:"APP_CRASH_TYPE_BREAKPAD"};S("report_client_error_with_app_crash_ks")||(mn.systemHealth={crashData:{clientError:{logMessage:{message:O.message}}}});mr("appCrashed",mn)}rr++}else"WARNING"===Tb&&Fk.ab("handleWarning",O);if(S("kevlar_gel_error_routing"))a:{var td=Tb;if(S("errors_via_jspb")){if(wr())Yh=void 0;else{var Bc=new kj;K(Bc,1,O.stack);O.fileName&&
K(Bc,4,O.fileName);var hb=O.lineNumber&&O.lineNumber.split?O.lineNumber.split(":"):[];0!==hb.length&&(1!==hb.length||isNaN(Number(hb[0]))?2!==hb.length||isNaN(Number(hb[0]))||isNaN(Number(hb[1]))||(ee(Bc,2,Number(hb[0])),ee(Bc,3,Number(hb[1]))):ee(Bc,2,Number(hb[0])));var Ub=new nj;K(Ub,1,O.message);K(Ub,3,O.name);ee(Ub,6,O.sampleWeight);"ERROR"===td?fe(Ub,2,2):"WARNING"===td?fe(Ub,2,1):fe(Ub,2,0);var $h=new lj;Sd($h,1,rd(!0));Zd($h,kj,3,mj,Bc);var Vb=new jj;K(Vb,3,window.location.href);for(var nn=
R("FEXP_EXPERIMENTS",[]),ai=0;ai<nn.length;ai++){var bi=void 0,ci=Vb.s,di=nn[ai],ud=ad(ci);od(ud);var on=ud&2,Fa=Rd(ci,ud,5);Array.isArray(Fa)||(Fa=md);var pn=!!(ud&32),ib=Yc(Fa);0===ib&&pn&&!on?(ib|=33,H(Fa,ib)):ib&1||(ib|=1,H(Fa,ib));if(on)ib&2||Wc(Fa,34),Object.freeze(Fa);else if(2&ib||2048&ib){Fa=Uc(Fa);var qn=1;pn&&(qn|=32);H(Fa,qn);Td(ci,ud,5,Fa)}bi=Fa;var rn=Yc(bi);di=yd(di,!!(4&rn)&&!!(4096&rn));bi.push(di)}var ei=Ej();if(!Fj()&&ei)for(var sn=w(Object.keys(ei)),Wb=sn.next();!Wb.done;Wb=sn.next()){var tn=
Wb.value,fi=new ij;K(fi,1,tn);K(fi,2,String(ei[tn]));ae(Vb,4,ij,fi)}var gi=O.params;if(gi){var un=w(Object.keys(gi));for(Wb=un.next();!Wb.done;Wb=un.next()){var vn=Wb.value,hi=new ij;K(hi,1,"client."+vn);K(hi,2,String(gi[vn]));ae(Vb,4,ij,hi)}}var wn=R("SERVER_NAME"),xn=R("SERVER_VERSION");if(wn&&xn){var ii=new ij;K(ii,1,"server.name");K(ii,2,wn);ae(Vb,4,ij,ii);var ji=new ij;K(ji,1,"server.version");K(ji,2,xn);ae(Vb,4,ij,ji)}var Ue=new oj;J(Ue,jj,1,Vb);J(Ue,lj,2,$h);J(Ue,nj,3,Ub);Yh=Ue}var yn=Yh;if(!yn)break a;
var zn=S("jspb_sparse_encoded_pivot")?new uj([{}]):new uj;Zd(zn,oj,163,vj,yn);or("clientError",zn)}else{var Pa=void 0;Pa=void 0===Pa?{}:Pa;if(wr())Xh=void 0;else{var vd={stackTrace:O.stack};O.fileName&&(vd.filename=O.fileName);var jb=O.lineNumber&&O.lineNumber.split?O.lineNumber.split(":"):[];0!==jb.length&&(1!==jb.length||isNaN(Number(jb[0]))?2!==jb.length||isNaN(Number(jb[0]))||isNaN(Number(jb[1]))||(vd.lineNumber=Number(jb[0]),vd.columnNumber=Number(jb[1])):vd.lineNumber=Number(jb[0]));var ki=
{level:"ERROR_LEVEL_UNKNOWN",message:O.message,errorClassName:O.name,sampleWeight:O.sampleWeight};"ERROR"===td?ki.level="ERROR_LEVEL_ERROR":"WARNING"===td&&(ki.level="ERROR_LEVEL_WARNNING");var fs={isObfuscated:!0,browserStackInfo:vd};Pa.pageUrl=window.location.href;Pa.kvPairs=[];R("FEXP_EXPERIMENTS")&&(Pa.experimentIds=R("FEXP_EXPERIMENTS"));var li=Ej();if(!Fj()&&li)for(var An=w(Object.keys(li)),Xb=An.next();!Xb.done;Xb=An.next()){var Bn=Xb.value;Pa.kvPairs.push({key:Bn,value:String(li[Bn])})}var mi=
O.params;if(mi){var Cn=w(Object.keys(mi));for(Xb=Cn.next();!Xb.done;Xb=Cn.next()){var Dn=Xb.value;Pa.kvPairs.push({key:"client."+Dn,value:String(mi[Dn])})}}var En=R("SERVER_NAME"),Fn=R("SERVER_VERSION");En&&Fn&&(Pa.kvPairs.push({key:"server.name",value:En}),Pa.kvPairs.push({key:"server.version",value:Fn}));Xh={errorMetadata:Pa,stackTrace:fs,logMessage:ki}}var Gn=Xh;if(!Gn)break a;mr("clientError",Gn)}if("ERROR"===td||S("errors_flush_gel_always_killswitch"))b:{if(S("web_fp_via_jspb")){var Ve=!0;Ve=
void 0===Ve?!1:Ve;var Hn=lr;lr=[];if(Hn)for(var In=w(Hn),ni=In.next();!ni.done;ni=In.next()){var Cc=ni.value;Ve?ir(Cc.Ac,Cc.payload,Fp,Cc.options):mr(Cc.Ac,Cc.payload,Cc.options)}kr(!0);if(!S("web_fp_via_jspb_and_json"))break b}kr()}}S("suppress_error_204_logging")||xr(Tb,O);vr(O)}}}}}else throw zc;}else console&&console.warn&&console.warn("The YouTube player is not attached to the DOM. API calls should be made after the onReady event. See more: https://developers.google.com/youtube/iframe_api_reference#Events")};function Ur(a){return(0===a.search("cue")||0===a.search("load"))&&"loadModule"!==a}
function Vr(a){return 0===a.search("get")||0===a.search("is")}
;function Wr(a,b){Nr.call(this,a,Object.assign({title:"video player",videoId:"",width:640,height:360},b||{}),"player");this.ka={};this.playerInfo={};this.videoTitle=""}
y(Wr,Nr);r=Wr.prototype;r.kc=function(){var a=Ir(this.h,"playerVars");if(a){var b={},c;for(c in a)b[c]=a[c];a=b}else a={};window!==window.top&&document.referrer&&(a.widget_referrer=document.referrer.substring(0,256));if(c=Ir(this.h,"embedConfig")){if(Oa(c))try{c=JSON.stringify(c)}catch(d){console.error("Invalid embed config JSON",d)}a.embed_config=c}return a};
r.fc=function(a){var b=a.event;a=a.info;switch(b){case "apiInfoDelivery":if(Oa(a))for(var c in a)a.hasOwnProperty(c)&&(this.ka[c]=a[c]);break;case "infoDelivery":Xr(this,a);break;case "initialDelivery":Oa(a)&&(window.clearInterval(this.i),this.playerInfo={},this.ka={},Yr(this,a.apiInterface),Xr(this,a));break;default:Rr(this,b,a)}};
function Xr(a,b){if(Oa(b)){for(var c in b)b.hasOwnProperty(c)&&(a.playerInfo[c]=b[c]);a.playerInfo.hasOwnProperty("videoData")&&(b=a.playerInfo.videoData,b.hasOwnProperty("title")&&b.title?(b=b.title,b!==a.videoTitle&&(a.videoTitle=b,a.g.setAttribute("title",b))):(a.videoTitle="",a.g.setAttribute("title","YouTube "+Ir(a.h,"title"))))}}
function Yr(a,b){db(b,function(c){this[c]||("getCurrentTime"===c?this[c]=function(){var d=this.playerInfo.currentTime;if(1===this.playerInfo.playerState){var e=(Date.now()/1E3-this.playerInfo.currentTimeLastUpdated_)*this.playerInfo.playbackRate;0<e&&(d+=Math.min(e,1))}return d}:Ur(c)?this[c]=function(){this.playerInfo={};
this.ka={};Tr(this,c,arguments);return this}:Vr(c)?this[c]=function(){var d=0;
0===c.search("get")?d=3:0===c.search("is")&&(d=2);return this.playerInfo[c.charAt(d).toLowerCase()+c.substr(d+1)]}:this[c]=function(){Tr(this,c,arguments);
return this})},a)}
r.getVideoEmbedCode=function(){var a=Ir(this.h,"host")+("/embed/"+Ir(this.h,"videoId")),b=Number(Ir(this.h,"width")),c=Number(Ir(this.h,"height"));if(isNaN(b)||isNaN(c))throw Error("Invalid width or height property");b=Math.floor(b);c=Math.floor(c);var d=this.videoTitle;a=Zb(a);d=Zb(null!=d?d:"YouTube video player");return'<iframe width="'+b+'" height="'+c+'" src="'+a+'" title="'+(d+'" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>')};
r.getOptions=function(a){return this.ka.namespaces?a?this.ka[a]?this.ka[a].options||[]:[]:this.ka.namespaces||[]:[]};
r.getOption=function(a,b){if(this.ka.namespaces&&a&&b&&this.ka[a])return this.ka[a][b]};
function gs(a){if("iframe"!==a.tagName.toLowerCase()){var b=Fr(a,"videoid");b&&(b={videoId:b,width:Fr(a,"width"),height:Fr(a,"height")},new Wr(a,b))}}
;F("YT.PlayerState.UNSTARTED",-1);F("YT.PlayerState.ENDED",0);F("YT.PlayerState.PLAYING",1);F("YT.PlayerState.PAUSED",2);F("YT.PlayerState.BUFFERING",3);F("YT.PlayerState.CUED",5);F("YT.get",function(a){return Br[a]});
F("YT.scan",Er);F("YT.subscribe",function(a,b,c){Tg.subscribe(a,b,c);Dr[a]=!0;for(var d in Br)Br.hasOwnProperty(d)&&Qr(Br[d],a)});
F("YT.unsubscribe",function(a,b,c){Sg(a,b,c)});
F("YT.Player",Wr);Nr.prototype.destroy=Nr.prototype.destroy;Nr.prototype.setSize=Nr.prototype.setSize;Nr.prototype.getIframe=Nr.prototype.getIframe;Nr.prototype.addEventListener=Nr.prototype.addEventListener;Wr.prototype.getVideoEmbedCode=Wr.prototype.getVideoEmbedCode;Wr.prototype.getOptions=Wr.prototype.getOptions;Wr.prototype.getOption=Wr.prototype.getOption;
Cr.push(function(a){var b=a;b||(b=document);a=gb(b.getElementsByTagName("yt:player"));var c=b||document;if(c.querySelectorAll&&c.querySelector)b=c.querySelectorAll(".yt-player");else{var d;c=document;b=b||c;if(b.querySelectorAll&&b.querySelector)b=b.querySelectorAll(".yt-player");else if(b.getElementsByClassName){var e=b.getElementsByClassName("yt-player");b=e}else{e=b.getElementsByTagName("*");var f={};for(c=d=0;b=e[c];c++){var g=b.className,h;if(h="function"==typeof g.split)h=0<=cb(g.split(/\s+/),
"yt-player");h&&(f[d++]=b)}f.length=d;b=f}}b=gb(b);db(fb(a,b),gs)});
"undefined"!=typeof YTConfig&&YTConfig.parsetags&&"onload"!=YTConfig.parsetags||Er();var hs=D.onYTReady;hs&&hs();var is=D.onYouTubeIframeAPIReady;is&&is();var js=D.onYouTubePlayerAPIReady;js&&js();}).call(this);
