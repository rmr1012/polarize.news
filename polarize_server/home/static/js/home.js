
var topWords= {"trump":null,"goverment shutdown":null,"muller investigation":null,"travewar":null,"border":null,"immigration":null,"abortion":null,"supreme court":null,"police":null,"south china sea":null,"democrats":null,"republicans":null,"syria":null,"mexico":null,"senate":null,"white house":null,"california":null,"election":null,"africa":null,"education":null,"nancy pelosi":null,"campaign":null};

$('#search').autocomplete({
    data:topWords,
    onAutocomplete: function(val){
          console.log(val);
      }
  });

var stuck=false;
var loadSense=500;
var loading=false;
$(".small-title").hide();
$(".small-title").css("opacity",0);
$(".spin-loader").css("opacity",0)

$(window).scroll(animate_stick);
$(document).ready(function(){
    $('.tabs').tabs(); // { swipeable: true }
    $('.lazy').Lazy();

  });
globalLoadCount=0
articleQuery="headlines"






function appendNewCard(){
          $.ajax({
                  type: "POST",
                  url: "load",//other option is search
                  dataType: "json",
                  data : {topic : topicList},
                  success: function(response) {
                      console.log(response);
                      $(".card-rack").append(response.card)
                      topicList.push(response.topic);
                      loading=false;
                      $(".spin-loader").css("opacity",0)
                  },
                  error: function(response) {
                      console.log(response);
                  }
          });
};


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
  else if ($(window).scrollTop()-($('.card-rack').offset().top+$('.card-rack').height())+$(window).height()+loadSense > 0 & !loading){
    loading=true;
    $(".spin-loader").css("opacity",1)
    console.log("starting to load");
    appendNewCard();
  }

}
