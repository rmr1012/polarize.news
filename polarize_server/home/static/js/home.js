
var topWords= {"trump":null,"president":null,"government":null,"shutdown":null,"border":null,"donald":null,"christmas":null,"washington":null,"police":null,"national":null,"democrats":null,"republicans":null,"syria":null,"security":null,"senate":null,"china":null,"california":null,"election":null,"war":null,"school":null,"pelosi":null,"campaign":null};

$('#search').autocomplete({
    data:topWords
  });

var stuck=false;
$(".small-title").hide();
$(".small-title").css("opacity",0);

$(window).scroll(animate_stick);
$(document).ready(function(){
    $('.tabs').tabs(); // { swipeable: true }
    $('.lazy').Lazy();

  });
globalLoadCount=0
articleQuery="headlines"


function makePlaceholder(){
  return "<div class='row lazy' id='box-"+globalLoadCount+"' data-loader='newsLoader'> <div class='col s6 libral' style='text-align: center;'>"
+"<img src='static/img/loader.svg'>"
+"</div>"
+"<div class='col s6 conservative' style='text-align: center;'>"
+"<img src='static/img/loader.svg'>"
+"</div>"
+"</div>"
+"<script> $('#box-"+globalLoadCount+"').Lazy();</script>"
}

(function($) {
       $.Lazy('newsLoader', function(element, response) {
           // just for demonstration, write some text inside element
           // element.find(".libral h4").html('lib successfully loaded div#' + element.attr('id'))
           //        .addClass("loaded");
           // element.find(".conservative h4").html('lib successfully loaded div#' + element.attr('id'))
           //       .addClass("loaded");
          console.log('lib successfully loaded div#' + element.attr('id'))
          globalLoadCount+=1;


          $.ajax({
                  type: "POST",
                  url: "load",//other option is search
                  dataType: "json",
                  data : { "topic":articleQuery, "index":globalLoadCount},
                  success: function(response) {
                      console.log(response);
                      element.html(response.card)
                  },
                  error: function(response) {
                      console.log(response);
                  }
          });
          setTimeout(function () { // just dummy delay for now
            element.append(makePlaceholder)
          }, 2000);

           // return loaded state to Lazy
           response(true);
       });
   })(window.jQuery || window.Zepto);

function animate_stick(){
  if ($(window).scrollTop()-$('.card-rack').offset().top+120 >0 & !stuck){
    stuck=true;
    console.log("stuck");
    $('nav').addClass('s8').removeClass('s12');
    setTimeout(function() {
      console.log("calling left toggle");
      $(".small-title").show();

    }, 500);
    setTimeout(function() {
      console.log("calling left toggle 2");
      $(".small-title").css("opacity",1);
    }, 600);




  }
  else if ($(window).scrollTop()-$('.card-rack').offset().top+120 <0 & stuck){
    stuck=false;
    console.log("un stuck");
    $(".small-title").css("opacity",0);
    setTimeout(function() {
      console.log("calling right toggle");
      $(".small-title").hide();
      $('nav').addClass('s12').removeClass('s8');
    }, 500);
  }

}
